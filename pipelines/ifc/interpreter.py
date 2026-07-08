from __future__ import annotations

import re
import time

import jupyter_client
import tiktoken

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Token budget for a single `python_exec` tool output. 8192 tokens
# (cl100k_base) is ~32kB of plain text -- large enough to show meaningful
# structure or a traceback, small enough that a single verbose call cannot
# dominate MiniMax M2.7's 196k-token context window even after 15-20 tool
# calls.
_DEFAULT_MAX_OUTPUT_TOKENS = 8192

# Split budget 70/30 between head and tail, so the agent sees both the
# beginning (schema, first entities printed) and the end (summary counts,
# final error lines) of an overly verbose output.
_HEAD_FRACTION = 0.70

# Module-level encoder -- `get_encoding` has a small I/O cost on first call.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _truncate_by_tokens(text: str, budget: int) -> str:
    """Truncate ``text`` to at most ``budget`` tokens (cl100k_base), keeping
    a head+tail slice so both the start and end of a long output stay visible.

    If the output fits within the budget, returns it unchanged. Otherwise
    appends a clear message instructing the agent to narrow its query.
    """
    ids = _ENCODING.encode(text)
    total = len(ids)
    if total <= budget:
        return text

    head_n = max(1, int(budget * _HEAD_FRACTION))
    tail_n = max(1, budget - head_n)
    dropped = total - head_n - tail_n

    head = _ENCODING.decode(ids[:head_n])
    tail = _ENCODING.decode(ids[-tail_n:])

    marker = (
        f"\n\n... [OUTPUT TRUNCATED. {dropped} of {total} tokens omitted "
        f"(budget: {budget} tokens). The previous tool call produced too much "
        f"text -- narrow your next query: slice the list, aggregate before "
        f"printing, or print only the fields you need.] ...\n\n"
    )
    return head + marker + tail


class JupyterInterpreter:
    """Persistent Python interpreter backed by a Jupyter kernel."""

    def __init__(self, max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS) -> None:
        self.max_output_tokens = max_output_tokens
        self.km = jupyter_client.KernelManager()
        self.km.start_kernel()
        self.kc = self.km.blocking_client()
        self.kc.wait_for_ready(timeout=30)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, code: str, timeout: int = 60) -> str:
        """Execute *code* in the kernel and return collected output."""
        msg_id = self.kc.execute(code)
        output_parts: list[str] = []
        deadline = time.monotonic() + timeout

        try:
            # Drain shell channel to detect execute_reply.
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError

                try:
                    iopub_msg = self.kc.get_iopub_msg(timeout=remaining)
                except Exception:
                    # Channel closed or kernel died.
                    break

                # Only inspect messages belonging to our execution.
                if iopub_msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = iopub_msg["header"]["msg_type"]
                content = iopub_msg["content"]

                if msg_type == "stream":
                    output_parts.append(content.get("text", ""))

                elif msg_type == "execute_result":
                    data = content.get("data", {})
                    output_parts.append(data.get("text/plain", ""))

                elif msg_type == "error":
                    tb = "\n".join(content.get("traceback", []))
                    output_parts.append(_ANSI_RE.sub("", tb))

                elif msg_type == "status" and content.get("execution_state") == "idle":
                    # Kernel finished processing our request.
                    break

        except TimeoutError:
            self.interrupt()
            return f"[Timeout] Execution exceeded {timeout}s and was interrupted."

        result = "".join(output_parts)
        return _truncate_by_tokens(result, self.max_output_tokens)

    # ------------------------------------------------------------------
    # Kernel lifecycle helpers
    # ------------------------------------------------------------------

    def interrupt(self) -> None:
        """Send SIGINT to the kernel."""
        try:
            self.km.interrupt_kernel()
        except Exception:
            pass

    def reset(self) -> None:
        """Restart the kernel with a clean namespace."""
        self.km.restart_kernel(now=True)
        self.kc.wait_for_ready(timeout=30)

    def shutdown(self) -> None:
        """Shut down the kernel immediately."""
        try:
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
