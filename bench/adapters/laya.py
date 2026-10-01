"""Adapter for the Laya decision models, run through the laya package's own Agent on CPU.

Two checkpoints go through it: the English one at the root of
convaiinnovations/laya and convaiinnovations/laya-multilingual. Each is loaded
directly with laya.load (the model cards' "single-model mode"), not through the
package's Router, so a row's model is the checkpoint it names. Read from the
cards and from laya 0.3.20's agent.py and common.py on 2026-09-27.

What goes in: the item's state and its questions exactly as HakemBench has them.
Agent.predict takes the typed-decision request shape as it is: choice criteria
as option to description (a null description means none), score criteria as a
list, noul criteria keyed true and false.

What the package does, as shipped, and nothing of ours on top:
- one encoder row per question, all of an item's questions in one forward pass;
- the checkpoint's own token budget (max_len, head_max_len in
  rl_agent_config.json); a state longer than what is left is cut from the right
  and an option text is cut to 48 tokens, both inside laya.common.build_sequence;
- the checkpoint's own temperatures per question type and option count, which the
  package clamps to 0.5 to 5 (the English checkpoint ships them, the multilingual
  one ships 1.0);
- probabilities and scores rounded to four places by the package.

What comes back: the package's answer fields. confidence on choice and score is
the package's own (1 minus the normalised entropy), so its source is "model"; its
max-probability confidence (answer_confidence) is kept in the diagnostics, next
to the raw logits before any temperature, read with a forward hook that changes
nothing.

A question the package cannot encode (its options overflow head_max_len) makes
the item unanswerable; `unanswerable` finds that by tokenising only, before any
forward pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from bench.adapters.base import AdapterError, AdapterResult, ModelInfo
from bench.adapters.decision_common import FROM_MODEL, as_text, runtime_versions, wire_questions
from schema.api import Request, Response
from schema.questions import ChoiceQuestion, NoulQuestion, Question

PACKAGE_VERSION = "0.3.20"
# The files laya.load itself downloads for a checkpoint at a repo root.
CHECKPOINT_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


@dataclass(frozen=True)
class Pin:
    repo: str
    revision: str
    what: str


CHECKPOINTS = {
    "laya": Pin("convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
                "English checkpoint (ModernBERT-large) at the repo root"),
    "laya-multilingual": Pin("convaiinnovations/laya-multilingual",
                             "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
                             "multilingual checkpoint (mmBERT-base)"),
}  # fmt: skip


def load_agent(pin: Pin, device: str) -> Any:
    """The checkpoint at the pinned revision, loaded by the pinned package."""
    import laya
    from huggingface_hub import snapshot_download

    if laya.__version__ != PACKAGE_VERSION:
        raise AdapterError(
            f"laya {laya.__version__} is installed, the rows expect {PACKAGE_VERSION}"
        )
    path = snapshot_download(pin.repo, revision=pin.revision, allow_patterns=CHECKPOINT_FILES)
    # laya.load takes a local directory as it is; given a repo id it would fetch the
    # latest revision, which a pinned run must not.
    return laya.load(path, device=device)


class LayaAdapter:
    """Answers a whole request with one Agent.predict call."""

    def __init__(self, name: str = "laya", *, agent: Any | None = None,
                 device: str = "cpu", max_len: int | None = None,
                 head_max_len: int | None = None) -> None:  # fmt: skip
        if name not in CHECKPOINTS:
            raise AdapterError(f"unknown Laya checkpoint {name!r}; known: {sorted(CHECKPOINTS)}")
        self.name = name
        self.pin = CHECKPOINTS[name]
        self.device = device
        self._agent = agent
        self._logits: Any = None
        self._hooked = False
        # The package's per-call overrides of the checkpoint's token budget (Agent.predict and
        # _encode_state take them); None keeps the shipped budget (the full-input rows).
        self.max_len = max_len
        self.head_max_len = head_max_len

    def _ready(self) -> Any:
        if self._agent is None:
            self._agent = load_agent(self.pin, self.device)
        model = getattr(self._agent, "model", None)
        if not self._hooked and model is not None and hasattr(model, "register_forward_hook"):
            model.register_forward_hook(self._keep_logits)
        self._hooked = True
        return self._agent

    def _keep_logits(self, module: Any, inputs: Any, output: Any) -> None:
        self._logits = output[0].detach().float().cpu().numpy()

    def _encoded(self, agent: Any, request: Request) -> tuple[list[str], list[dict]]:
        """The package's own validation and tokenisation of every question, no forward pass."""
        wire = wire_questions(request.questions)
        ids = list(wire)
        for qid in ids:
            agent._check_question(qid, wire[qid])
        internal = {qid: agent._to_internal(wire[qid]) for qid in ids}
        return ids, agent._encode_state(request.state, ids, internal, max_len=self.max_len,
                                        head_max_len=self.head_max_len)  # fmt: skip

    def unanswerable(self, request: Request) -> str | None:
        """Why the package cannot take this request, or None when it can."""
        try:
            self._encoded(self._ready(), request)
        except ValueError as exc:
            return str(exc)
        return None

    def answer(self, request: Request) -> AdapterResult:
        agent = self._ready()
        ids, encoded = self._encoded(agent, request)
        wire = wire_questions(request.questions)

        self._logits = None
        started = time.perf_counter()
        result = agent.predict(request.state, wire, max_len=self.max_len,
                               head_max_len=self.head_max_len)  # fmt: skip
        latency_ms = (time.perf_counter() - started) * 1000

        answers: dict[str, Any] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        confidence_source: dict[str, str] = {}
        state_tokens = _state_tokens(agent, request.state)
        for row, qid in enumerate(ids):
            question = request.questions[qid]
            given = result["answers"][qid]
            answers[qid] = _typed(question, given)
            diag: dict[str, Any] = {
                "answer_confidence": given.get("answer_confidence"),
                "act_probability": (given.get("action") or {}).get("act_probability"),
            }
            if state_tokens is not None:
                diag["state_tokens"] = state_tokens
                diag["state_tokens_read"] = _state_tokens_read(agent, encoded[row])
            if self._logits is not None:
                k = len(encoded[row]["markers"])
                diag["logits"] = [float(v) for v in self._logits[row, :k]]
            diagnostics[qid] = diag
            if not isinstance(question, NoulQuestion):
                confidence_source[qid] = FROM_MODEL

        usage = result.get("usage")
        response = Response.model_validate(
            {"model": self.pin.repo, "answers": answers, "usage": usage}
        )
        return AdapterResult(response=response, latency_ms=latency_ms, per_question_ms=None,
                             confidence_source=confidence_source,
                             diagnostics=diagnostics)  # fmt: skip

    def _budget(self, cfg: dict) -> str:
        return _budget_text(cfg, self.max_len, self.head_max_len)

    def info(self) -> ModelInfo:
        agent = self._ready()
        cfg = getattr(agent, "cfg", {}) or {}
        temperature = "shipped temperatures as the package applies them"
        if cfg.get("temperature_by_options"):
            temperature += " (per question type and option count)"
        return ModelInfo(
            adapter="laya",
            model=self.pin.repo,
            revision=self.pin.revision,
            path_used=(
                f"laya {PACKAGE_VERSION} Agent.predict on the {self.pin.what}, loaded directly "
                f"(no Router), fp32; {self._budget(cfg)}; {temperature}; " + runtime_versions()
            ),
            device=self.device,
            prompt_version=None,
        )


def _budget_text(cfg: dict, max_len: int | None, head_max_len: int | None) -> str:
    if max_len is None and head_max_len is None:
        return f"max_len {cfg.get('max_len')}, head_max_len {cfg.get('head_max_len')} as shipped"
    return (f"max_len {max_len or cfg.get('max_len')}, head_max_len "
            f"{head_max_len or cfg.get('head_max_len')} by the package's per-call override, "
            f"for full input (shipped {cfg.get('max_len')} and "
            f"{cfg.get('head_max_len')})")  # fmt: skip


def _typed(question: Question, given: dict[str, Any]) -> dict[str, Any]:
    """The package's answer in the contract's shape, its own numbers unchanged."""
    if given.get("type") != question.type:
        raise AdapterError(f"asked a {question.type} question, the package answered "
                           f"{given.get('type')!r}")  # fmt: skip
    if isinstance(question, NoulQuestion):
        return {"type": "noul", "noul": given["noul"]}
    if isinstance(question, ChoiceQuestion):
        return {"type": "choice", "choice": given["choice"],
                "probabilities": given["probabilities"],
                "confidence": given["confidence"]}  # fmt: skip
    return {"type": "score", "score": given["score"], "legend": given["legend"],
            "probabilities": given["probabilities"],
            "confidence": given["confidence"]}  # fmt: skip


def _state_tokens(agent: Any, state: Any) -> int | None:
    """The state's length in the checkpoint's tokens, as the package tokenises it."""
    tok = getattr(agent, "tok", None)
    if tok is None:
        return None
    # as_text is laya.common.serialize_state: text as it is, JSON with ensure_ascii off.
    text = as_text(state).replace(tok.mask_token, " ")
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _state_tokens_read(agent: Any, row: dict) -> int:
    """How many state tokens the row kept: [CLS] head [SEP] options [SEP] state [SEP]."""
    ids, markers = row["ids"], row["markers"]
    options_end = ids.index(agent.tok.sep_token_id, markers[-1])
    return len(ids) - options_end - 2
