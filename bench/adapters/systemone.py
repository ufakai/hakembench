"""Shared plumbing for open decision models that speak the typed-decision wire format.

decider, Kev and simple-jev each take a request as {"state", "questions"} in
the shape of the public contract (schema/api.py) and return one answer dict per
question. This module builds that body from a harness Request, reads the
answers back into schema.questions answers, and runs one item through a
model's own call. It adds nothing to the prompt and changes no probability:
each project's code renders the prompt, applies its own temperature and
rounds as it ships.

A question a model's own code refuses (its documented option, level or
length limits) raises Unanswerable, an AdapterError the driver that runs the
model counts and records instead of writing a row. Any other failure stops the
run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from bench.adapters.base import AdapterError, AdapterResult
from schema.api import Request, Response, Usage, check_response
from schema.questions import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

# The answer fields the contract defines; anything else a model returns goes to diagnostics.
CONTRACT_FIELDS = {"type", "choice", "noul", "score", "confidence", "probabilities", "legend"}
# decider, Kev and simple-jev return their own confidence on choice and score answers.
CONFIDENCE_SOURCE = "model"
# A probability-weighted level can pass the last level by float rounding;
# more than this is an error.
SCORE_SLACK = 1e-6


class Unanswerable(AdapterError):
    """The model's own code refused the item (a documented limit). Counted, never guessed."""


def body(request: Request) -> dict[str, Any]:
    """The request as the wire format has it: state and questions exactly as the item holds them.

    A field the item leaves unset (a noul without criteria) is left out rather
    than sent as null; a choice option without a description stays as null,
    as the contract asks.
    """
    dumped = request.model_dump(mode="json")
    questions = {}
    for qid, question in dumped["questions"].items():
        questions[qid] = {key: value for key, value in question.items() if value is not None}
    return {"state": dumped["state"], "questions": questions}


def to_answer(question: Question, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One returned answer dict as a schema answer, and the extra fields as diagnostics.

    Probabilities, choice, score and confidence are taken as the model returned
    them. The only change is clipping a score that passes the level range by a
    float rounding (at most SCORE_SLACK).
    """
    extra = {key: value for key, value in raw.items() if key not in CONTRACT_FIELDS}
    if raw.get("type") != question.type:
        raise AdapterError(f"asked a {question.type} question, got {raw.get('type')!r}")
    if isinstance(question, NoulQuestion):
        return {"type": "noul", "noul": float(raw["noul"])}, extra
    probabilities = {str(key): float(value) for key, value in raw["probabilities"].items()}
    if isinstance(question, ChoiceQuestion):
        if set(probabilities) != set(question.options):
            raise AdapterError("the returned probabilities do not cover the options exactly")
        answer = {
            "type": "choice",
            "choice": str(raw["choice"]),
            "probabilities": {key: probabilities[key] for key in question.options},
            "confidence": float(raw["confidence"]),
        }
        return answer, extra
    assert isinstance(question, ScoreQuestion)
    top = len(question.criteria) - 1.0
    score = float(raw["score"])
    if not -SCORE_SLACK <= score <= top + SCORE_SLACK:
        raise AdapterError(f"score {score} lies outside 0 to {top:g}")
    levels = [str(k) for k in range(len(question.criteria))]
    if set(probabilities) != set(levels):
        raise AdapterError("the returned probabilities do not cover the levels exactly")
    answer = {
        "type": "score",
        "score": min(max(score, 0.0), top),
        "legend": dict(zip(levels, question.criteria, strict=True)),
        "probabilities": {key: probabilities[key] for key in levels},
        "confidence": float(raw["confidence"]),
    }
    if "legend" in raw and {str(k): v for k, v in raw["legend"].items()} != answer["legend"]:
        extra["returned_legend"] = raw["legend"]
    return answer, extra


def answer_with(
    request: Request,
    call: Callable[[dict[str, Any]], dict[str, Any]],
    model_id: str,
    confidence: str = CONFIDENCE_SOURCE,
) -> AdapterResult:
    """Run one item through `call` (the model's own entry point) and read its response.

    `call` takes the wire body and returns the model's response dict with an
    "answers" map and, when the model counts them, "usage".
    """
    started = time.perf_counter()
    raw = call(body(request))
    latency_ms = (time.perf_counter() - started) * 1000
    answers: dict[str, Any] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    confidence_source: dict[str, str] = {}
    returned = raw.get("answers", {})
    for qid, question in request.questions.items():
        if qid not in returned:
            raise AdapterError(f"the model returned no answer for {qid}")
        answers[qid], diagnostics[qid] = to_answer(question, returned[qid])
        if not isinstance(question, NoulQuestion):
            confidence_source[qid] = confidence
    usage = raw.get("usage") or {}
    # No text is generated, so no output tokens; the input count is the model's own.
    tokens = usage.get("input_tokens")
    try:
        response = Response(
            model=model_id,
            answers=answers,
            usage=Usage(input_tokens=int(tokens), output_tokens=0) if tokens is not None else None,
        )
        check_response(request, response)
    except ValueError as error:
        raise AdapterError(f"the model's answer does not fit the question: {error}") from None
    return AdapterResult(
        response=response,
        latency_ms=latency_ms,
        per_question_ms=None,
        confidence_source=confidence_source,
        diagnostics=diagnostics,
    )


def describe(runtime: dict[str, Any]) -> str:
    """Plain words for path_used: the route, then every version and setting in `runtime`."""
    return "; ".join(f"{key} {value}" for key, value in runtime.items())
