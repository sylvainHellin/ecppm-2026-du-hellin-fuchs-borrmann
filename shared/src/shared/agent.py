"""Agent execution wrapper: run questions through a DeepAgent and collect results."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from openinference.instrumentation import get_attributes_from_context
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, format_span_id

_tracer = trace.get_tracer("shared.agent")


@dataclass
class AgentResult:
    """Normalized result from a single question run."""

    answer: str
    trace: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    num_tool_calls: int = 0
    elapsed_s: float = 0.0
    span_id: str = ""


def _parent_span_attrs(question: str) -> dict:
    attrs: dict = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
        SpanAttributes.INPUT_VALUE: question,
        SpanAttributes.INPUT_MIME_TYPE: "text/plain",
    }
    # Inherit any `using_metadata(...)` / session / user attributes set by callers.
    for key, value in get_attributes_from_context():
        attrs[key] = value
    return attrs


def answer_question(
    agent,
    question: str,
    *,
    recursion_limit: int = 120,
    verbose: bool = False,
) -> AgentResult:
    """Run a single question through a DeepAgent and return normalized results.

    Each call uses a fresh thread_id for a clean context. Token usage is
    extracted from AIMessage.usage_metadata across all messages.

    The whole invocation is wrapped in a single OTEL parent span named
    ``question`` (kind=AGENT). Every downstream LangChain/LangGraph span
    (ChatCompletion, tool calls, middleware) nests underneath, so the Phoenix
    waterfall shows one collapsible tree per question.
    """
    thread_id = str(uuid.uuid4())
    config: dict = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    # Propagate session.id from the OpenInference context (set by
    # using_session at the call site) into the LangGraph config metadata.
    # Without this the LangChain instrumentor falls back to thread_id as
    # session.id, creating one Phoenix session per question instead of
    # grouping all questions in a run under a single session.
    for key, value in get_attributes_from_context():
        if key == "session.id":
            config["metadata"] = {"session_id": value}
            break
    input_msg = {"messages": [{"role": "user", "content": question}]}

    t0 = time.perf_counter()

    with _tracer.start_as_current_span(
        "question",
        attributes=_parent_span_attrs(question),
    ) as span:
        hex_span_id = format_span_id(span.get_span_context().span_id)
        try:
            if not verbose:
                result = agent.invoke(input_msg, config=config)
                elapsed = time.perf_counter() - t0
                messages = result["messages"]
                answer = _normalize_content(messages[-1].content)
                trace_entries = _extract_trace(messages)
                input_tok, output_tok, tool_calls = _sum_usage(messages)
                agent_result = AgentResult(
                    answer=answer,
                    trace=trace_entries,
                    input_tokens=input_tok,
                    output_tokens=output_tok,
                    num_tool_calls=tool_calls,
                    elapsed_s=round(elapsed, 2),
                    span_id=hex_span_id,
                )
                _annotate_span(span, agent_result)
                return agent_result

            agent_result = _run_verbose(agent, input_msg, config, t0)
            agent_result.span_id = hex_span_id
            _annotate_span(span, agent_result)
            return agent_result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            raise


def _annotate_span(span, result: AgentResult) -> None:
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.answer)
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(result.input_tokens))
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(result.output_tokens))
    span.set_attribute(
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
        int(result.input_tokens + result.output_tokens),
    )
    span.set_attribute("tool.call_count", int(result.num_tool_calls))
    span.set_attribute("elapsed_s", float(result.elapsed_s))


def _run_verbose(agent, input_msg, config, t0) -> AgentResult:
    trace_entries: list[dict] = []
    step = 0
    final_content = ""
    input_tok = 0
    output_tok = 0
    tool_call_count = 0

    for chunk in agent.stream(input_msg, config=config, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        for _node_name, data in chunk.items():
            if not isinstance(data, dict):
                continue
            messages = data.get("messages", [])
            if hasattr(messages, "value"):
                messages = messages.value
            if not isinstance(messages, list):
                continue
            for msg in messages:
                msg_type = getattr(msg, "type", None)

                # Accumulate token usage from every AI message
                usage = getattr(msg, "usage_metadata", None)
                if usage and isinstance(usage, dict):
                    input_tok += usage.get("input_tokens", 0)
                    output_tok += usage.get("output_tokens", 0)

                if msg_type == "ai":
                    entry: dict = {"role": "assistant"}
                    content = _normalize_content(msg.content)
                    if content:
                        entry["content"] = content
                        preview = content[:500] + "..." if len(content) > 500 else content
                        print(f"    [thinking] {preview}")
                        final_content = content
                    tool_calls = getattr(msg, "tool_calls", [])
                    if tool_calls:
                        tc_list = []
                        for tc in tool_calls:
                            step += 1
                            tool_call_count += 1
                            tc_list.append({
                                "step": step,
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                            })
                            args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
                            if len(args_str) > 500:
                                args_str = args_str[:500] + "..."
                            print(f"    [{step}] tool: {tc.get('name', '?')}({args_str})")
                        entry["tool_calls"] = tc_list
                    if content or tool_calls:
                        trace_entries.append(entry)

                elif msg_type == "tool":
                    raw = msg.content if isinstance(msg.content, str) else str(msg.content)
                    trace_entries.append({
                        "role": "tool",
                        "name": getattr(msg, "name", ""),
                        "content": raw,
                    })
                    preview = raw[:500].replace("\n", " ")
                    if len(raw) > 500:
                        preview += "..."
                    print(f"         -> {preview}")

    elapsed = time.perf_counter() - t0
    return AgentResult(
        answer=final_content,
        trace=trace_entries,
        input_tokens=input_tok,
        output_tokens=output_tok,
        num_tool_calls=tool_call_count,
        elapsed_s=round(elapsed, 2),
    )


def _normalize_content(content) -> str:
    """Flatten LangChain message content (str or list of blocks) to a plain string."""
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        ).strip()
    return str(content).strip() if content else ""


def _extract_trace(messages) -> list[dict]:
    """Extract execution trace from LangGraph message history."""
    entries: list[dict] = []
    step = 0
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type == "human":
            continue
        elif msg_type == "ai":
            entry: dict = {"role": "assistant"}
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
                entries.append(entry)
        elif msg_type == "tool":
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            entries.append({
                "role": "tool",
                "name": getattr(msg, "name", ""),
                "content": raw,
            })
    return entries


def _sum_usage(messages) -> tuple[int, int, int]:
    """Sum token usage and tool call count across all messages."""
    input_tok = 0
    output_tok = 0
    tool_calls = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if usage and isinstance(usage, dict):
            input_tok += usage.get("input_tokens", 0)
            output_tok += usage.get("output_tokens", 0)
        tc = getattr(msg, "tool_calls", None)
        if tc:
            tool_calls += len(tc)
    return input_tok, output_tok, tool_calls
