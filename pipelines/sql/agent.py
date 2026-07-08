"""SQL agent factory: DeepAgent wired with a persistent python_exec tool and pre-connected SQLite."""

from __future__ import annotations

from typing import Any, Callable

from deepagents import create_deep_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from shared import init_llm

from interpreter import SqlInterpreter
from prompts import render_sql_prompt

_WRAP_UP_MSG = (
    "You have reached the exploration limit. Write your final answer now. "
    "Follow the answer format from your instructions: "
    "cite the source table and column/property for every value in parentheses, "
    "state any assumptions with 'Assuming [condition], [conclusion]', "
    "and if the information is not in the model, say so explicitly. "
    "IMPORTANT: If information was not found, report it as a factual finding "
    "('The model contains no X'), NOT as 'I cannot determine'."
)


class RestrictToolsMiddleware(AgentMiddleware):
    """Hide every tool except an allow-listed set from the model.

    ``create_deep_agent`` unconditionally injects built-in tools (``ls``,
    ``glob``, ``read_file``, ``task``, ...) backed by an in-memory *virtual*
    filesystem (``StateBackend``). The system prompt tells the agent it has only
    ``python_exec``, so when ``python_exec`` briefly misbehaved the model
    "discovered" these tools, queried the virtual FS (which sees only ``/tmp``),
    "confirmed" the real database was missing, and hallucinated a final answer.
    Filtering the request's tool list keeps reality aligned with the prompt.
    """

    def __init__(self, *, allowed: set[str]) -> None:
        self._allowed = allowed

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | Any:
        tools = getattr(request, "tools", None) or []
        filtered = [t for t in tools if getattr(t, "name", None) in self._allowed]
        if len(filtered) != len(tools):
            request = request.override(tools=filtered)
        return handler(request)


class RecursionGuardMiddleware(AgentMiddleware):
    """Force the agent to summarize when it has been running too long.

    Counts AI messages in state as a proxy for agent steps. When the count
    exceeds ``warn_after``, tools are stripped from the request and a user
    message is injected asking the agent to wrap up. This forces a text-only
    response, which naturally ends the agent loop.
    """

    def __init__(self, *, warn_after: int = 40) -> None:
        self._warn_after = warn_after

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | Any:
        ai_count = sum(
            1 for m in request.state.get("messages", [])
            if getattr(m, "type", None) == "ai"
        )
        if ai_count >= self._warn_after:
            request = request.override(
                tools=[],
                messages=[*request.messages, HumanMessage(content=_WRAP_UP_MSG)],
            )
        return handler(request)


def create_sql_agent(
    model: str,
    db_path: str,
    *,
    max_retries: int = 3,
    warn_after: int = 25,
):
    """Create a DeepAgent that queries a SQLite database via iterative Python execution.

    The agent is given a single ``python_exec`` tool backed by a persistent Jupyter
    kernel with a pre-connected ``sqlite3`` database (``conn`` and ``cursor`` are
    available immediately). The system prompt (rendered from ``system_prompt.jinja2``)
    includes the full database schema so the agent starts informed.

    A ``RecursionGuardMiddleware`` is installed to warn the agent after
    ``warn_after`` AI messages, giving it a chance to wrap up before hitting
    the hard recursion limit.

    The returned agent exposes the interpreter as ``agent._sql_interpreter``, so
    callers can ``reset()`` the kernel between questions (which also re-injects
    the database connection) while reusing the same compiled graph.
    """
    llm = init_llm(model, temperature=0, max_retries=max_retries)
    interp = SqlInterpreter()
    interp.setup_db(db_path)

    @tool
    def python_exec(code: str) -> str:
        """Execute Python code in a persistent Jupyter kernel.

        Variables, imports, and loaded data persist across calls within a single
        question. A SQLite connection is pre-established: use conn and cursor
        directly. Only stdout and stderr are returned -- use print() to see results.
        """
        return interp.run(code)

    agent = create_deep_agent(
        model=llm,
        tools=[python_exec],
        system_prompt=render_sql_prompt(db_path),
        middleware=[
            RestrictToolsMiddleware(allowed={"python_exec"}),
            RecursionGuardMiddleware(warn_after=warn_after),
        ],
    )
    agent._sql_interpreter = interp
    return agent
