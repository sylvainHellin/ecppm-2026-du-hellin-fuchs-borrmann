"""Phoenix (Arize) OTEL tracing setup for LangChain/LangGraph agents."""

from __future__ import annotations

from openinference.instrumentation import using_metadata, using_session  # noqa: F401 -- re-export
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register


def init_tracing(project_name: str = "bim-query-pipelines"):
    """Initialize Phoenix OTEL tracing. Call once at process start.

    The Phoenix server must already be running (see README: 'Running the IFC pipeline').

    `register(auto_instrument=True)` is unreliable about picking up LangChain in
    practice, so we explicitly instantiate `LangChainInstrumentor` against the
    tracer provider returned by `register`.
    """
    tracer_provider = register(project_name=project_name)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
