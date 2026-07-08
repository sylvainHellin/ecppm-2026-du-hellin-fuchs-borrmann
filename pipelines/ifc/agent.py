"""IFC agent factory: DeepAgent wired with a persistent python_exec tool."""

from __future__ import annotations

from typing import Any, Callable

from deepagents import create_deep_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from shared import init_llm

from interpreter import JupyterInterpreter
from prompts import render_ifc_prompt

_WRAP_UP_MSG = (
    "You have reached the exploration limit. Write your final answer now. "
    "Follow the answer format from your instructions: "
    "cite the IFC source for every value in parentheses, "
    "state any assumptions with 'Assuming [condition], [conclusion]', "
    "and if the information is not in the model, say so explicitly."
)


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


def create_ifc_agent(
    model: str,
    ifc_path: str,
    *,
    max_retries: int = 3,
    warn_after: int = 25,
):
    """Create a DeepAgent that explores an IFC model via iterative Python execution.

    The agent is given a single `python_exec` tool backed by a persistent Jupyter
    kernel. The system prompt (rendered from `system_prompt.jinja2`) instructs the
    agent to open the model itself via `ifcopenshell.open(ifc_path)` on first use.
    Variables, imports, and loaded data persist across tool calls within a single
    question.

    A ``RecursionGuardMiddleware`` is installed to warn the agent after
    ``warn_after`` AI messages (roughly ``warn_after * 2`` LangGraph supersteps),
    giving it a chance to wrap up before hitting the hard recursion limit.

    The returned agent exposes the interpreter as `agent._ifc_interpreter`, so
    callers can `reset()` the kernel between questions to get a clean namespace
    while reusing the same compiled graph.
    """
    llm = init_llm(model, temperature=0, max_retries=max_retries)
    interp = JupyterInterpreter()

    @tool
    def python_exec(code: str) -> str:
        """Execute Python code in a persistent Jupyter kernel.

        Variables, imports, and loaded data persist across calls within a single
        question. Only stdout and stderr are returned -- use print() to see results.
        """
        return interp.run(code)

    agent = create_deep_agent(
        model=llm,
        tools=[python_exec],
        system_prompt=render_ifc_prompt(ifc_path),
        middleware=[RecursionGuardMiddleware(warn_after=warn_after)],
    )
    agent._ifc_interpreter = interp
    return agent
