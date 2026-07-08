#!/usr/bin/env python3
"""
Neo4j Cypher Agent — A deepagents-based agent that answers BIM questions by
querying an IFC model stored as a Neo4j property graph (imported by ifc2neo4j).

Uses a single ``execute_cypher`` tool with a read-only Neo4j user.
Filesystem middleware tools are stripped; TodoList, SubAgent, and
Summarization are preserved.

Usage:
    from agent import create_neo4j_agent, answer_question_neo4j

    agent = create_neo4j_agent("openai:gpt-4.1", "bolt://localhost:7687",
                               neo4j_password="bench_admin")
    answer = answer_question_neo4j(agent, "How many walls are in the building?")
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from deepagents import create_deep_agent
import deepagents.middleware.filesystem as _fs_middleware

# ── Strip ALL filesystem tools (agent only uses execute_cypher) ────────────
_FS_TOOLS = ("ls", "read_file", "glob", "grep", "execute", "write_file", "edit_file")
_original_fs_init = _fs_middleware.FilesystemMiddleware.__init__


def _patched_fs_init(self, **kwargs):
    _original_fs_init(self, **kwargs)
    self.tools = [t for t in self.tools if t.name not in _FS_TOOLS]
    self._custom_system_prompt = ""


_fs_middleware.FilesystemMiddleware.__init__ = _patched_fs_init

# ── Global driver reference (one per process) ─────────────────────────────
_neo4j_driver = None


def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        raise RuntimeError("Neo4j driver not initialised — call create_neo4j_agent first")
    return _neo4j_driver


def _run_read_query(driver, query: str) -> list[dict]:
    """Execute *query* inside a Neo4j **read transaction**.

    The database will reject any write operation (CREATE, DELETE, SET, …)
    at the transaction level — no regex needed.
    """
    with driver.session() as session:
        return session.execute_read(lambda tx: list(tx.run(query)))


def _make_execute_cypher_tool():
    """Build the execute_cypher tool that closes over the global driver."""

    @tool
    def execute_cypher(query: str) -> str:
        """Execute a read-only Cypher query against the IFC graph in Neo4j.

        Returns query results as formatted text (one row per line).
        Only read queries are allowed — write operations will be rejected
        by the database.

        Args:
            query: A Cypher query string (MATCH, RETURN, WITH, etc.)
        """
        driver = _get_driver()
        try:
            records = _run_read_query(driver, query)
        except Exception as e:
            return f"ERROR: {e}"

        if not records:
            return "(no results)"

        keys = records[0].keys()
        lines = ["\t".join(keys)]
        for rec in records[:500]:
            lines.append("\t".join(_format_value(rec[k]) for k in keys))
        result = "\n".join(lines)
        if len(records) > 500:
            result += f"\n... ({len(records)} total rows, showing first 500)"
        return result

    return execute_cypher


def _format_value(val) -> str:
    if val is None:
        return "null"
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False, default=str)
    return str(val)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.jinja2"


def render_system_prompt(**kwargs) -> str:
    """Load and render this agent's system prompt from system_prompt.jinja2.

    Pass template variables as keyword arguments (e.g. ``ifc_filename=...``).
    """
    from jinja2 import Template

    return Template(_PROMPT_PATH.read_text(encoding="utf-8")).render(**kwargs)


# ---------------------------------------------------------------------------
# Data classes and trace helpers (shared with agent.py)
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Result from answer_question_neo4j, including the final answer and execution trace."""
    answer: str
    trace: list[dict] = field(default_factory=list)


def _normalize_content(content) -> str:
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        ).strip()
    return str(content).strip() if content else ""


def _extract_trace(messages) -> list[dict]:
    """Extract execution trace from LangGraph message history."""
    trace = []
    step = 0
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type == "human":
            continue
        elif msg_type == "ai":
            entry = {"role": "assistant"}
            content = _normalize_content(msg.content)
            if content:
                entry["content"] = content
            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                tc_list = []
                for tc in tool_calls:
                    step += 1
                    tc_list.append({
                        "step": step,
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    })
                entry["tool_calls"] = tc_list
            if content or tool_calls:
                trace.append(entry)
        elif msg_type == "tool":
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            trace.append({
                "role": "tool",
                "name": getattr(msg, "name", ""),
                "content": raw,
            })
    return trace


# ---------------------------------------------------------------------------
# LLM initialisation (shared logic with agent.py)
# ---------------------------------------------------------------------------

def _init_llm(model: str, **kwargs):
    """Initialize a LangChain chat model, with shortcuts for GLM, Gemini, Grok, MiniMax."""
    # LangChain passes request_timeout=None to the OpenAI SDK, which the SDK
    # interprets as "disable all timeouts" instead of "use defaults".  Set an
    # explicit timeout to prevent infinite hangs on stalled API connections.
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("timeout", httpx.Timeout(600.0, connect=10.0))
    if model.startswith("glm:"):
        model_name = model.split(":", 1)[1]
        return init_chat_model(
            f"openai:{model_name}",
            base_url="https://api.z.ai/api/coding/paas/v4",
            api_key=os.environ.get("Z_AI_API_KEY", ""),
            **kwargs,
        )
    if model.startswith("gemini:"):
        model_name = model.split(":", 1)[1]
        return init_chat_model(
            f"google_genai:{model_name}",
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            **kwargs,
        )
    if model.startswith("grok:"):
        model_name = model.split(":", 1)[1]
        return init_chat_model(
            f"openai:{model_name}",
            base_url="https://api.x.ai/v1",
            api_key=os.environ.get("XAI_API_KEY", ""),
            **kwargs,
        )
    if model.startswith("minimax:"):
        model_name = model.split(":", 1)[1]
        kwargs.setdefault("max_tokens", 16384)
        return init_chat_model(
            f"openai:{model_name}",
            base_url="https://api.minimax.io/v1",
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            **kwargs,
        )
    return init_chat_model(model, **kwargs)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def create_neo4j_agent(
    model: str,
    neo4j_uri: str = "bolt://localhost:7687",
    *,
    neo4j_user: str = "neo4j",
    neo4j_password: str = "bench_admin",
    max_retries: int = 5,
    system_prompt: str | None = None,
):
    """Create a deepagents agent that queries an IFC Neo4j graph.

    Neo4j Community Edition only supports a single user, so the agent
    connects with the same credentials used for import.  Write safety is
    enforced by the keyword blocklist in ``execute_cypher``.

    Args:
        model: Model identifier (e.g., "openai:gpt-4.1", "glm:glm-4.7")
        neo4j_uri: Neo4j bolt URI
        neo4j_user: Neo4j user
        neo4j_password: Neo4j password
        max_retries: Max retries for rate-limit / transient API errors

    Returns:
        A compiled LangGraph StateGraph ready for invocation
    """
    import neo4j as neo4j_lib

    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
    _neo4j_driver = neo4j_lib.GraphDatabase.driver(
        neo4j_uri, auth=(neo4j_user, neo4j_password),
    )
    _neo4j_driver.verify_connectivity()

    llm = _init_llm(model, temperature=0, max_retries=max_retries)

    cypher_tool = _make_execute_cypher_tool()

    agent = create_deep_agent(
        model=llm,
        tools=[cypher_tool],
        system_prompt=system_prompt or render_system_prompt(),
    )

    return agent


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------

def answer_question_neo4j(
    agent,
    question: str,
    *,
    recursion_limit: int = 120,
    verbose: bool = False,
) -> AgentResult:
    """Run a single question through the Neo4j agent.

    Each call uses a fresh thread_id for a clean context.

    Args:
        agent: A compiled agent from create_neo4j_agent()
        question: The question to answer about the IFC model
        recursion_limit: Maximum agent loop iterations
        verbose: If True, print each tool call and result in real time

    Returns:
        AgentResult with the final answer and full execution trace
    """
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }

    if not verbose:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )
        answer = _normalize_content(result["messages"][-1].content)
        trace = _extract_trace(result["messages"])
        return AgentResult(answer=answer, trace=trace)

    # Verbose: stream for real-time output
    trace = []
    step = 0
    final_content = ""

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        if not isinstance(chunk, dict):
            continue
        for node_name, data in chunk.items():
            if not isinstance(data, dict):
                continue
            messages = data.get("messages", [])
            if hasattr(messages, "value"):
                messages = messages.value
            if not isinstance(messages, list):
                continue
            for msg in messages:
                msg_type = getattr(msg, "type", None)

                if msg_type == "ai":
                    entry = {"role": "assistant"}
                    content = _normalize_content(msg.content)
                    if content:
                        entry["content"] = content
                        preview = content[:500] + "…" if len(content) > 500 else content
                        print(f"    [thinking] {preview}")
                        final_content = content
                    tool_calls = getattr(msg, "tool_calls", [])
                    if tool_calls:
                        tc_list = []
                        for tc in tool_calls:
                            step += 1
                            tc_list.append({
                                "step": step,
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                            })
                            args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
                            if len(args_str) > 500:
                                args_str = args_str[:500] + "…"
                            print(f"    [{step}] tool: {tc.get('name', '?')}({args_str})")
                        entry["tool_calls"] = tc_list
                    if content or tool_calls:
                        trace.append(entry)

                elif msg_type == "tool":
                    raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                    trace.append({
                        "role": "tool",
                        "name": getattr(msg, "name", ""),
                        "content": raw,
                    })
                    preview = raw[:500].replace("\n", " ")
                    if len(raw) > 500:
                        preview += "…"
                    print(f"         → {preview}")

    return AgentResult(answer=final_content, trace=trace)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ask a question about an IFC model in Neo4j")
    ap.add_argument("question", help="The question to ask")
    ap.add_argument("--model", default="openai:gpt-4.1", help="LLM model (default: openai:gpt-4.1)")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j bolt URI")
    ap.add_argument("--neo4j-user", default="neo4j", help="Neo4j user")
    ap.add_argument("--neo4j-password", default="bench_admin", help="Neo4j password")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print agent steps")
    ap.add_argument("--trace-output", "-t", default=None, help="Save execution trace to JSON file")
    args = ap.parse_args()

    agent = create_neo4j_agent(
        args.model,
        args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )
    result = answer_question_neo4j(agent, args.question, verbose=args.verbose)
    print(result.answer)

    if args.trace_output:
        trace_data = {"question": args.question, "answer": result.answer, "trace": result.trace}
        with open(args.trace_output, "w") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nTrace saved to {args.trace_output}")
