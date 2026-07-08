"""Render the IFC system prompt from its Jinja template."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Template

_TEMPLATE_PATH = Path(__file__).parent / "system_prompt.jinja2"


def render_ifc_prompt(ifc_path: str) -> str:
    """Render the IFC system prompt for a given model path."""
    template = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.render(ifc_path=ifc_path)
