"""
Veneers Package

Modular veneer generation architecture with clean separation of concerns.
"""

from .pipeline import VeneerPipeline
from .presets import get_preset, interpolate_presets, PRESETS
from .prompt_templates import get_prompts, list_templates, PROMPT_TEMPLATES

__all__ = [
    "VeneerPipeline",
    "get_preset",
    "interpolate_presets",
    "PRESETS",
    "get_prompts",
    "list_templates",
    "PROMPT_TEMPLATES",
]
