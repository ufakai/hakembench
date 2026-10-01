"""What every adapter looks like to the harness.

An adapter takes a schema.api.Request and returns a schema.api.Response for
it, with timing. The harness does not know whether the answer came from a
remote decision API, a hosted chat-completions endpoint or weights on this
machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from schema.api import Request, Response


class AdapterError(RuntimeError):
    """The adapter could not produce a valid answer. Never carries a credential."""


@dataclass(frozen=True)
class ModelInfo:
    adapter: str  # "typesafe", "chat_completions", "local" or "karar"
    model: str  # the model id as the adapter was given it
    revision: str | None  # weights commit or file hash, when the adapter can know it
    path_used: str  # plain words: which route produced the answers
    device: str | None = None  # local adapters only
    prompt_version: str | None = None  # label-reading adapters only


@dataclass
class AdapterResult:
    response: Response
    # Wall time of the whole call, in milliseconds.
    latency_ms: float
    # Set when the adapter answers each question with its own call.
    per_question_ms: dict[str, float] | None = None
    # How each confidence was produced, per question id:
    # "model" when the source returned it, "max_probability" when we derived it.
    confidence_source: dict[str, str] = field(default_factory=dict)
    # Anything worth keeping next to the measurement, per question id.
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)


class Adapter(Protocol):
    def info(self) -> ModelInfo: ...

    def answer(self, request: Request) -> AdapterResult: ...
