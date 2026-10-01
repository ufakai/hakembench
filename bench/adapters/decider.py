"""decider (github.com/Mapika/decider): a Qwen3.5 base with a head that reads the answer slot.

The published inference path is `decider.infer.Decider(path).system_one(state,
questions)`: on CUDA in bf16 with its CUDA-graph engine, the layout,
temperatures (one per answer type) and isolated score levels its
decider_config.json sets, every question scored in its own row. Its answers
come back rounded to four places, as the package returns them; this adapter
keeps them as they are. Its confidence on choice and score answers is the
package's own (the typed-decision contract's definition), so the rows carry
confidence_source "model".

decider's own code refuses a question outside its limits (2 to 255 options, 2
to 10 levels) with a ValueError; that item is Unanswerable. A state longer
than its 32,768-token budget is cut by its own code, as it documents.

decider is imported only in load(), so this module imports without torch;
tests inject a fake with a system_one method.
"""

from __future__ import annotations

from typing import Any, Protocol

from bench.adapters.base import AdapterResult, ModelInfo
from bench.adapters.systemone import Unanswerable, answer_with, describe
from schema.api import Request

ADAPTER = "decider"


class SystemOne(Protocol):
    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]: ...


def load(model_dir: str, device: str = "cuda") -> tuple[SystemOne, dict[str, Any]]:
    """The Decider on a downloaded snapshot directory, with its settings as loaded."""
    from decider.infer import Decider

    decider = Decider(model_dir, device=device)
    settings = {
        "layout": decider.layout,
        "temperature": decider.T,
        "temperature_by_type": decider.T_by_type,
        "isolated_levels": decider.isolated_levels,
        "schema_first": decider.schema_first,
        "neutralize_none": decider.neutralize_none,
        "cuda_graphs": decider.eng is not None,
        "dtype": str(next(decider.m.parameters()).dtype).removeprefix("torch."),
        "version": decider.name,
    }
    return decider, settings


class DeciderAdapter:
    """One item, one Decider.system_one call."""

    def __init__(
        self,
        decider: SystemOne,
        *,
        model_id: str,
        revision: str | None,
        runtime: dict[str, Any],
        device: str | None = None,
    ) -> None:
        self._decider = decider
        self.model_id = model_id
        self.revision = revision
        self.runtime = runtime
        self.device = device

    def _call(self, wire: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._decider.system_one(wire["state"], wire["questions"])
        except ValueError as error:
            raise Unanswerable(f"decider refused the item: {error}") from None

    def answer(self, request: Request) -> AdapterResult:
        return answer_with(request, self._call, self.model_id)

    def info(self) -> ModelInfo:
        return ModelInfo(
            adapter=ADAPTER,
            model=self.model_id,
            revision=self.revision,
            path_used=describe(
                {
                    "route": "decider.infer.Decider.system_one, its own temperatures as shipped",
                    **self.runtime,
                }
            ),
            device=self.device,
            prompt_version=None,
        )
