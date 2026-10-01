"""Kev (github.com/jaredpalmer/kev): a LoRA adapter and a pointer head on a Qwen3.5 base.

The published inference path is the one Kev's own serving endpoint runs
(skills/kev-deploy/scripts/kev_serve.py): `Checkpoint(run).load("cuda",
LoadOptions(dtype=bfloat16, cuda_graphs=True, fused=True))`, a `kev.serve.Server`
on it, the same three warm-up requests, and `Server.answer` for each request,
which is what its POST /v1/systemone returns. A model whose CUDA graphs and fused
kernels do not fit the GPU runs with both off (`eager=True`), which LoadOptions
documents as the path Kev's own reported numbers use and kev.serve exposes as
KEV_CUDA_GRAPHS=0 and KEV_FUSED=0; the weights stay bf16. The temperature the checkpoint
carries is applied by Kev's own head as it loads, as shipped. Answers come back
rounded to four places by Kev's code and are kept as they are; its confidence
on choice and score answers is Kev's own, so the rows carry confidence_source
"model".

Kev's own code refuses a request it cannot take (its request schema, or a state
over its serving limits) with a validation error or an HTTP 422; that item is
Unanswerable.

kev is imported only in load(), so this module imports without torch; tests
inject a fake server with an answer method.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench.adapters.base import AdapterResult, ModelInfo
from bench.adapters.systemone import Unanswerable, answer_with, describe
from schema.api import Request

ADAPTER = "kev"
# The name Kev's server answers to (kev.serve.MODEL_NAMES); it selects nothing.
SERVED_NAME = "kev-latest"
# kev_serve.py's warm-up: its example questions on a short, a medium and a long ticket, in English.
# They capture CUDA graphs before the first item; no item is part of them.
_QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery, delays",
                                "billing": "Charges, payments"}},
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}  # fmt: skip
_TICKET = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. "
WARMUP = [{"state": _TICKET * n, "model": SERVED_NAME, "questions": _QUESTIONS} for n in (1, 4, 16)]


def load(
    run: str, device: str = "cuda", eager: bool = False
) -> tuple[Callable[[dict[str, Any]], dict], dict]:
    """Kev's serving path on `run` (a Hub id, `@revision` allowed), warmed up as kev_serve.py does.

    Returns the call that answers one wire body, and the settings as loaded.
    """
    import torch
    from kev.api import SystemOneRequest
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.serve import Server

    checkpoint = Checkpoint(run)
    tokenizer, model = checkpoint.load(
        device, LoadOptions(dtype=torch.bfloat16, cuda_graphs=not eager, fused=not eager)
    )
    server = Server(checkpoint, tokenizer, model, device)
    for request in WARMUP:
        server.answer(SystemOneRequest.model_validate(request))
    server.wait_idle()

    def call(wire: dict[str, Any]) -> dict[str, Any]:
        request = SystemOneRequest.model_validate({**wire, "model": SERVED_NAME})
        return server.answer(request)

    meta = checkpoint.meta
    settings = {
        "temperature": float(model.head.temperature),
        "base": meta.base,
        "base_revision": meta.base_revision,
        "lora_rank": meta.lora,
        "backend": model.backend,
        "dtype": str(model.dtype).removeprefix("torch."),
        "cuda_graphs": getattr(model, "graphs", None) is not None,
        "fused_kernels": not eager,
    }
    return call, settings


def _refused(error: Exception) -> bool:
    """A validation error from Kev's request schema, or the 422 its server raises for a limit."""
    return isinstance(error, ValueError) or getattr(error, "status_code", None) == 422


class KevAdapter:
    """One item, one Kev Server.answer call."""

    def __init__(
        self,
        call: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        model_id: str,
        revision: str | None,
        runtime: dict[str, Any],
        device: str | None = None,
    ) -> None:
        self._call_model = call
        self.model_id = model_id
        self.revision = revision
        self.runtime = runtime
        self.device = device

    def _call(self, wire: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._call_model(wire)
        except Exception as error:
            if _refused(error):
                raise Unanswerable(f"Kev refused the item: {error}") from None
            raise

    def answer(self, request: Request) -> AdapterResult:
        return answer_with(request, self._call, self.model_id)

    def info(self) -> ModelInfo:
        return ModelInfo(
            adapter=ADAPTER,
            model=self.model_id,
            revision=self.revision,
            path_used=describe(
                {
                    "route": "kev.serve.Server.answer as kev_serve.py loads it, "
                    "the checkpoint's own temperature as shipped",
                    **self.runtime,
                }
            ),
            device=self.device,
            prompt_version=None,
        )
