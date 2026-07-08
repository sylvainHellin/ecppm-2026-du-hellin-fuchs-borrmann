"""Persistent Python interpreter with a pre-connected SQLite database.

Reuses the same Jupyter-kernel approach as the IFC pipeline's interpreter.py
(process isolation, real SIGINT on interrupt, token-truncated output). Adds
``setup_db(db_path)`` to inject a ``sqlite3`` connection into the kernel
namespace so the agent's ``python_exec`` calls can use ``conn`` and ``cursor``
directly.
"""

from __future__ import annotations

import re
import time

import jupyter_client
import tiktoken

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_DEFAULT_MAX_OUTPUT_TOKENS = 8192
_HEAD_FRACTION = 0.70
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _truncate_by_tokens(text: str, budget: int) -> str:
    """Truncate ``text`` to at most ``budget`` tokens (cl100k_base), keeping
    a head+tail slice so both the start and end of a long output stay visible.
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


# An alive kernel echoes a `status: busy` / `execute_input` on iopub within
# milliseconds of `execute()`. If nothing arrives for this long, the kernel is
# dead or its iopub channel has desynced from the client (the classic
# jupyter_client pitfall after a restart) -- treat it as unresponsive rather
# than silently returning empty output.
_ACK_TIMEOUT = 10

_KERNEL_DEAD_MSG = (
    "[KERNEL ERROR] The Python kernel is not responding and could not be "
    "recovered after a restart. This is an environment failure, NOT a data "
    "problem -- do not conclude that the database or any file is missing. "
    "Stop retrying and report the kernel failure."
)


class JupyterInterpreter:
    """Persistent Python interpreter backed by a Jupyter kernel.

    Hardened against the failure mode where a kernel dies or its iopub channel
    desyncs from the client: such a kernel previously returned empty output
    silently (every ``print`` looked like a no-op), which let the agent
    hallucinate "the file does not exist" and burn hours retrying. Now ``run``
    detects an unresponsive kernel, self-heals once via a hard ``reset``, and
    surfaces an explicit error if recovery fails.
    """

    def __init__(self, max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS) -> None:
        self.max_output_tokens = max_output_tokens
        # The blocking client MUST be created here, before any asyncio event
        # loop is running (i.e. before the agent starts invoking). A blocking
        # client created later, under a running loop, has ``get_iopub_msg``
        # return un-awaited coroutines instead of blocking -- which spins the
        # read loop and wedges the kernel. So we create it once and reuse it
        # across restarts (see ``reset``), never rebuilding it mid-run.
        self.km = jupyter_client.KernelManager()
        self.km.start_kernel()
        self.kc = self.km.blocking_client()
        self.kc.wait_for_ready(timeout=30)

    def _execute(self, code: str, timeout: int) -> tuple[str, bool]:
        """Low-level execute. Returns ``(output, alive)``.

        ``alive`` is ``False`` only when the kernel never acknowledged the
        request (no iopub traffic within ``_ACK_TIMEOUT``) -- i.e. it is dead or
        desynced. A slow-but-alive kernel that overruns ``timeout`` is
        interrupted and reported as a timeout with ``alive=True``. Does NOT
        self-heal, so it is safe to call from recovery paths.
        """
        msg_id = self.kc.execute(code)
        output_parts: list[str] = []
        start = time.monotonic()
        deadline = start + timeout
        acked = False

        while True:
            now = time.monotonic()
            if not acked and now - start > _ACK_TIMEOUT:
                return "".join(output_parts), False
            remaining = deadline - now
            if remaining <= 0:
                self.interrupt()
                output_parts.append(
                    f"\n[Timeout] Execution exceeded {timeout}s and was interrupted."
                )
                return "".join(output_parts), True

            try:
                iopub_msg = self.kc.get_iopub_msg(timeout=min(remaining, _ACK_TIMEOUT))
            except Exception:
                # No message within the poll window; loop re-checks ack/deadline.
                continue

            if iopub_msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            acked = True
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
                break

        return "".join(output_parts), True

    def run(self, code: str, timeout: int = 60) -> str:
        """Execute *code* and return its (token-truncated) output.

        If the kernel is unresponsive the code never ran, so it is safe to hard
        ``reset`` and retry once. A second failure is reported explicitly rather
        than masquerading as empty output.
        """
        output, alive = self._execute(code, timeout)
        if not alive:
            self.reset()
            output, alive = self._execute(code, timeout)
            if not alive:
                return _KERNEL_DEAD_MSG
        return _truncate_by_tokens(output, self.max_output_tokens)

    def interrupt(self) -> None:
        try:
            self.km.interrupt_kernel()
        except Exception:
            pass

    def reset(self) -> None:
        """Restart the kernel with a clean namespace, reusing the same client.

        Deliberately minimal: restart the kernel process (same connection/ports)
        and wait for it, reusing the existing blocking client and its channels.
        We do NOT rebuild the client or its channels, because anything created
        under a running event loop returns coroutines from ``get_iopub_msg``
        instead of blocking, which wedges the read loop. If a restart still
        leaves the kernel unresponsive, ``run``'s liveness check surfaces an
        explicit error rather than silently returning empty output.
        """
        self.km.restart_kernel(now=True)
        self.kc.wait_for_ready(timeout=30)

    def shutdown(self) -> None:
        try:
            if self.kc is not None:
                self.kc.stop_channels()
        except Exception:
            pass
        try:
            if self.km is not None:
                self.km.shutdown_kernel(now=True)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass


class SqlInterpreter(JupyterInterpreter):
    """JupyterInterpreter with a pre-connected SQLite database.

    After ``setup_db(db_path)`` is called, the kernel has ``sqlite3``,
    ``conn``, and ``cursor`` available. ``reset()`` automatically re-injects
    the connection so the agent can keep using ``conn``/``cursor`` after a
    kernel restart.
    """

    def __init__(self, max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS) -> None:
        super().__init__(max_output_tokens=max_output_tokens)
        self._db_path: str | None = None

    def setup_db(self, db_path: str) -> None:
        """Inject a sqlite3 connection into the kernel namespace."""
        self._db_path = db_path
        self._inject_connection()

    def _inject_connection(self) -> None:
        """Run the startup code that establishes conn/cursor.

        Uses the low-level ``_execute`` (not ``run``) on purpose: ``run``
        self-heals by calling ``reset``, and ``reset`` calls this method, so
        going through ``run`` here would risk infinite recursion if the kernel
        stayed dead.
        """
        if self._db_path is None:
            return
        # Escape backslashes and quotes for safe string interpolation
        escaped = self._db_path.replace("\\", "\\\\").replace("'", "\\'")
        setup_code = (
            "import sqlite3\n"
            f"conn = sqlite3.connect('{escaped}')\n"
            "conn.row_factory = sqlite3.Row\n"
            "cursor = conn.cursor()\n"
        )
        self._execute(setup_code, timeout=30)

    def reset(self) -> None:
        """Restart the kernel and re-inject the database connection."""
        super().reset()
        self._inject_connection()
