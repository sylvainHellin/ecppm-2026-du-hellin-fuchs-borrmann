#!/usr/bin/env python3
"""
IFC Agent — A deepagents-based agent that answers BIM questions by exploring
an IFC model's filesystem representation (produced by ifc2fs.py).

Uses FilesystemBackend with built-in tools (ls, read_file, glob, grep)
to navigate the converted IFC filesystem structure.

Usage:
    from agent import create_ifc_agent, answer_question

    agent = create_ifc_agent("glm:glm-4.7", "/path/to/project/arc_fs")
    answer = answer_question(agent, "How many walls are in the building?")
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent

import httpx

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellBackend
import deepagents.middleware.filesystem as _fs_middleware

_fs_middleware.DEFAULT_READ_LIMIT = 2000
_fs_middleware.READ_FILE_TOOL_DESCRIPTION = """\
Reads a file from the filesystem.

Usage:
- By default, it reads up to 2000 lines starting from the beginning of the file. \
For most element JSON files (30-200 lines), you can omit the limit parameter.
- For very large files, use offset and limit for pagination.
- Results are returned with line numbers starting at 1.
- You can call multiple read_file in a single response to batch-read files."""

# ── Remove write_file / edit_file tools (agent is read-only + execute) ────
_fs_middleware.FILESYSTEM_SYSTEM_PROMPT = """\
## Tool Usage and File Reading

Follow the tool docs for the available tools. In particular, for filesystem tools, \
use pagination (offset/limit) when reading large files.

## Filesystem Tools `ls`, `read_file`, `glob`, `grep`

You have access to a read-only filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- glob: find files matching a pattern (e.g., "**/*.json")
- grep: search for text within files

Do NOT create or modify files. Use `execute('python3 -c "..."')` for all computation."""

_WRITE_TOOLS = ("write_file", "edit_file")
_original_fs_init = _fs_middleware.FilesystemMiddleware.__init__


def _patched_fs_init(self, **kwargs):
    _original_fs_init(self, **kwargs)
    self.tools = [t for t in self.tools if t.name not in _WRITE_TOOLS]


_fs_middleware.FilesystemMiddleware.__init__ = _patched_fs_init

def snapshot_fs(fs_root: str | Path) -> frozenset[str]:
    """Return the set of top-level item names in an _fs directory."""
    root = Path(fs_root)
    if not root.is_dir():
        return frozenset()
    return frozenset(item.name for item in root.iterdir())


_DATA_PREFIXES = ("Site__", "__", ".")


def _initial_cleanup(fs_root: str | Path) -> None:
    """Best-effort cleanup for a dirty _fs directory when no snapshot exists.

    Used once in ``create_ifc_agent`` before taking the snapshot, to handle
    leftovers from previous crashed runs or other agents.  Keeps items whose
    names start with ``Site__``, ``__``, or ``.`` — everything produced by
    ``ifc2fs.py`` follows these conventions.
    """
    root = Path(fs_root)
    if not root.is_dir():
        return
    removed = []
    for item in root.iterdir():
        if any(item.name.startswith(p) for p in _DATA_PREFIXES):
            continue
        removed.append(item.name)
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)
    if removed:
        print(f"  [cleanup] Initial cleanup of {fs_root}: removed {removed}")


def cleanup_fs(fs_root: str | Path, original: frozenset[str]) -> None:
    """Restore an _fs directory to its original snapshot state.

    1. Deletes top-level entries that were added after the snapshot.
    2. Raises ``RuntimeError`` if any entry from the snapshot is missing
       (indicates the agent deleted data during execution).
    """
    root = Path(fs_root)
    if not root.is_dir():
        return

    current = {item.name for item in root.iterdir()}

    missing = original - current
    if missing:
        raise RuntimeError(
            f"Agent corrupted {fs_root}: the following original items were "
            f"deleted during execution: {sorted(missing)}"
        )

    added = current - original
    if added:
        print(f"  [cleanup] Removing {len(added)} leftover(s) from {fs_root}: {sorted(added)}")
    for name in added:
        target = root / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)


_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.jinja2"


def render_system_prompt(**kwargs) -> str:
    """Load and render this agent's system prompt from system_prompt.jinja2.

    Pass template variables as keyword arguments (e.g. ``ifc_filename=...``).
    """
    from jinja2 import Template

    return Template(_PROMPT_PATH.read_text(encoding="utf-8")).render(**kwargs)


@dataclass
class AgentResult:
    """Result from answer_question, including the final answer and execution trace."""
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


def _init_llm(model: str, **kwargs):
    """Initialize a LangChain chat model, with shortcuts for GLM and Gemini.

    Accepts standard LangChain model strings (e.g. "openai:gpt-4.1") plus:
      - "glm:<model>"     → Z.AI OpenAI-compatible endpoint
      - "gemini:<model>"  → Google GenAI
      - "grok:<model>"    → xAI OpenAI-compatible endpoint
      - "minimax:<model>" → MiniMax OpenAI-compatible endpoint
    """
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


def create_ifc_agent(model: str, fs_root: str, *, max_retries: int = 5,
                     system_prompt: str | None = None):
    """Create a deepagents agent scoped to a specific IFC filesystem directory.

    Args:
        model: Model identifier (e.g., "openai:gpt-4.1", "glm:glm-4.7",
               "gemini:gemini-3-pro-preview", "grok:grok-3",
               "minimax:MiniMax-M2.7",
               "anthropic:claude-sonnet-4-5-20250929")
        fs_root: Absolute path to the converted IFC filesystem directory
        max_retries: Max retries for rate-limit / transient API errors

    Returns:
        A compiled LangGraph StateGraph ready for invocation
    """
    fs_root = str(Path(fs_root).resolve())

    llm = _init_llm(model, temperature=0, max_retries=max_retries)

    # Clean up any leftovers from a previous run/crash, then snapshot.
    # Two-pass: first remove non-data items (heuristic), then snapshot the
    # guaranteed-clean state for precise per-question tracking.
    _initial_cleanup(fs_root)
    fs_snapshot = snapshot_fs(fs_root)

    venv_bin = _PIPELINE_DIR / ".venv" / "bin"
    backend = LocalShellBackend(
        root_dir=fs_root,
        virtual_mode=True,
        env={"PATH": f"{venv_bin}:/usr/bin:/bin"},
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=system_prompt or render_system_prompt(),
        backend=backend,
    )

    agent._ifc_fs_root = fs_root
    agent._ifc_fs_snapshot = fs_snapshot
    return agent


def answer_question(agent, question: str, *, recursion_limit: int = 120, verbose: bool = False) -> AgentResult:
    """Run a single question through the agent and return the answer with execution trace.

    Each call uses a fresh thread_id for a clean context.

    Args:
        agent: A compiled agent from create_ifc_agent()
        question: The question to answer about the IFC model
        recursion_limit: Maximum agent loop iterations (default 120)
        verbose: If True, print each tool call and result in real time

    Returns:
        AgentResult with the final answer and full execution trace
    """
    fs_root = getattr(agent, "_ifc_fs_root", None)
    fs_snapshot = getattr(agent, "_ifc_fs_snapshot", None)
    if fs_root is not None and fs_snapshot is not None:
        cleanup_fs(fs_root, fs_snapshot)

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

    # Verbose: stream for real-time output and collect trace simultaneously
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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ask a question about an IFC model")
    ap.add_argument("fs_root", help="Path to the converted IFC filesystem (e.g., projects/duplex/arc_fs)")
    ap.add_argument("question", help="The question to ask")
    ap.add_argument("--model", default="openai:gpt-5.3-codex", help="LLM model (default: openai:gpt-5.3-codex)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print agent steps")
    ap.add_argument("--trace-output", "-t", default=None, help="Save execution trace to JSON file")
    args = ap.parse_args()

    agent = create_ifc_agent(args.model, args.fs_root)
    result = answer_question(agent, args.question, verbose=args.verbose)
    print(result.answer)

    if args.trace_output:
        trace_data = {"question": args.question, "answer": result.answer, "trace": result.trace}
        with open(args.trace_output, "w") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nTrace saved to {args.trace_output}")
