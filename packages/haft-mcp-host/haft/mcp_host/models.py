"""Public value types for the MCP host chat loop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class McpServer:
    """Configuration for a single MCP server endpoint."""

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", dict(self.headers))


@dataclass(slots=True)
class Turn:
    """A completed logical user turn and its raw debug artifacts."""

    user_prompt: str
    response_text: str
    output_items: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    round_trips: list[dict[str, Any]] = field(default_factory=list)
