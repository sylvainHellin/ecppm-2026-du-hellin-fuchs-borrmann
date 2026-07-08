"""LLM initialization with model prefix routing."""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model


def init_llm(model: str, *, request_timeout: float | None = 120, **kwargs):
    """Initialize a LangChain chat model with prefix-based provider routing.

    Args:
        model: Prefixed model id (e.g. ``minimax:MiniMax-M2.7``).
        request_timeout: HTTP request timeout in seconds. Passed as ``request_timeout``
            to OpenAI-compatible providers, ``timeout`` to Gemini/Anthropic, and
            forwarded via kwargs to anything else. ``None`` disables the timeout.
        **kwargs: Extra keyword arguments forwarded to ``init_chat_model()``.

    Supported prefixes:
      minimax:<model>    MiniMax PAYG (OpenAI-compatible)
      fireworks:<model>  Fireworks AI (OpenAI-compatible)
      glm:<model>        Z.AI / GLM (OpenAI-compatible)
      gemini:<model>     Google GenAI
      grok:<model>       xAI (OpenAI-compatible)
      openai:<model>     OpenAI (pass-through)
      anthropic:<model>  Anthropic (pass-through)
    """
    prefix, _, model_name = model.partition(":")
    if not model_name:
        # No prefix -- pass the whole string to LangChain
        if request_timeout is not None:
            kwargs.setdefault("timeout", request_timeout)
        return init_chat_model(model, **kwargs)

    openai_compat = {
        "minimax": ("https://api.minimax.io/v1", "MINIMAX_API_KEY"),
        "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
        "glm": ("https://api.z.ai/api/coding/paas/v4", "Z_AI_API_KEY"),
        "grok": ("https://api.x.ai/v1", "XAI_API_KEY"),
    }

    if prefix in openai_compat:
        base_url, env_key = openai_compat[prefix]
        if request_timeout is not None:
            kwargs.setdefault("request_timeout", request_timeout)
        return init_chat_model(
            f"openai:{model_name}",
            base_url=base_url,
            api_key=os.environ.get(env_key, ""),
            **kwargs,
        )

    if prefix == "gemini":
        if request_timeout is not None:
            kwargs.setdefault("timeout", request_timeout)
        return init_chat_model(
            f"google_genai:{model_name}",
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            **kwargs,
        )

    # openai, anthropic, or any other LangChain-native prefix
    if request_timeout is not None:
        # openai uses request_timeout; anthropic/others use timeout
        if prefix == "openai":
            kwargs.setdefault("request_timeout", request_timeout)
        else:
            kwargs.setdefault("timeout", request_timeout)
    return init_chat_model(model, **kwargs)
