"""Provider-neutral LLM interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]


class LLMProvider(Protocol):
    def complete(self, prompt: str, *, tools: Sequence[ToolDefinition] = ()) -> str:
        """Return a provider response for a prepared prompt and optional tools."""
        ...
