"""simple-jev (github.com/featherless-ai/simple-jev): next-token label logits from a chat model.

The published inference path is its HF server's service without the HTTP
layer: `hf_server.load_service(model, revision=..., device="cuda",
dtype="bfloat16")` with every other setting at the server's defaults, then
`DecisionService.classify(request)`, which is what its POST /v1/classifier and
/v1/systemone return. The prompt format is the one its loader selects for the
model's architecture and size when no format flag is given; the rows record it
as the prompt version. simple-jev ships no calibration and none is applied.
Its choice and score confidence is the largest label probability, returned by
its own code; that confidence is the plain top probability, so the rows carry
confidence_source "max_probability" (corrected after the run).

simple-jev's own code refuses a request over its limits (options, levels,
tokens per question branch) with a ValueError, never truncating; that item is
Unanswerable.

hf_server is imported only in load(), so this module imports without torch;
tests inject a fake service with an async classify method.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from bench.adapters.base import AdapterResult, ModelInfo
from bench.adapters.systemone import Unanswerable, answer_with, describe
from schema.api import Request

ADAPTER = "simple-jev"


class Service(Protocol):
    metadata: dict[str, Any]

    async def classify(self, request: Any) -> dict[str, Any]: ...


def load(model: str, revision: str, device: str = "cuda") -> tuple[Service, dict[str, Any]]:
    """The server's service for `model` at `revision`, with the server's defaults otherwise."""
    from hf_server import load_service

    service = load_service(model, revision=revision, device=device, dtype="bfloat16")
    settings = {
        "prompt_policy": service.metadata.get("prompt_policy"),
        "prompt_policy_selection": service.metadata.get("prompt_policy_selection", {}).get("mode"),
        "dtype": "bfloat16",
        "calibration": "none (simple-jev ships none)",
    }
    return service, settings


class SimpleJevAdapter:
    """One item, one DecisionService.classify call, on one event loop kept for the whole run."""

    def __init__(
        self,
        service: Service,
        *,
        model_id: str,
        revision: str | None,
        runtime: dict[str, Any],
        device: str | None = None,
    ) -> None:
        self._service = service
        self._loop = asyncio.new_event_loop()
        self.model_id = model_id
        self.revision = revision
        self.runtime = runtime
        self.device = device

    def _call(self, wire: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._loop.run_until_complete(
                self._service.classify({"model": self.model_id, **wire})
            )
        except ValueError as error:
            raise Unanswerable(f"simple-jev refused the item: {error}") from None

    def answer(self, request: Request) -> AdapterResult:
        # Its confidence is the plain top probability, not a trained signal.
        return answer_with(request, self._call, self.model_id, confidence="max_probability")

    def info(self) -> ModelInfo:
        policy = self._service.metadata.get("prompt_policy")
        return ModelInfo(
            adapter=ADAPTER,
            model=self.model_id,
            revision=self.revision,
            path_used=describe(
                {
                    "route": "hf_server.DecisionService.classify at the server's defaults, "
                    "no calibration",
                    **self.runtime,
                }
            ),
            device=self.device,
            prompt_version=f"simple-jev {policy}" if policy else None,
        )
