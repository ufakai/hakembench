"""Pieces the step 9 baseline adapters share (laya.py and openjev.py import them).

Each of those models is run through its own published inference code on this
machine. What they share is the edge: the request's questions written out as
JSON the way the typed-decision contract writes them, a typed answer built from
a probability distribution, and the version of the weights and of the package.
"""

from __future__ import annotations

import json
from typing import Any

from schema.questions import ChoiceQuestion, JsonContent, NoulQuestion, Question, expected_level

# How each confidence came about (bench/adapters/base.py AdapterResult).
FROM_MODEL = "model"
MAX_PROBABILITY = "max_probability"


def as_text(content: JsonContent) -> str:
    """A state or an instruction as one string: text as it is, JSON compact and unescaped."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def wire_question(question: Question) -> dict[str, Any]:
    """One question as the typed-decision request writes it.

    Optional fields left unset are left out rather than sent as null, and so is
    a noul criterion with no description. A choice option with no description
    keeps its null: the contract asks for one key per option.
    """
    body = question.model_dump(mode="json")
    for name in [name for name, value in body.items() if value is None]:
        del body[name]
    if body["type"] == "noul" and "criteria" in body:
        body["criteria"] = {k: v for k, v in body["criteria"].items() if v is not None}
        if not body["criteria"]:
            del body["criteria"]
    return body


def wire_questions(questions: dict[str, Question]) -> dict[str, dict[str, Any]]:
    return {qid: wire_question(q) for qid, q in questions.items()}


def outcome_keys(question: Question) -> list[str]:
    """The keys of an answer's distribution: options, level indices, or false then true."""
    if isinstance(question, ChoiceQuestion):
        return question.options
    if isinstance(question, NoulQuestion):
        return ["false", "true"]
    return [str(k) for k in range(len(question.criteria))]


def answer_from_distribution(question: Question, probabilities: list[float]) -> dict[str, Any]:
    """A typed answer from a distribution in outcome_keys order, confidence the maximum probability.

    For models whose own output is the distribution alone. The choice is the
    first most probable option, the score the probability-weighted level. The
    level is held inside the scale: a float32 distribution can sum a hair above
    one and push it past the last level by a rounding error.
    """
    keys = outcome_keys(question)
    if len(keys) != len(probabilities):
        raise ValueError(f"{len(probabilities)} probabilities for {len(keys)} outcomes")
    distribution = {k: float(p) for k, p in zip(keys, probabilities, strict=True)}
    if isinstance(question, NoulQuestion):
        return {"type": "noul", "noul": distribution["true"]}
    top = max(distribution.values())
    if isinstance(question, ChoiceQuestion):
        choice = next(k for k in keys if distribution[k] == top)
        return {"type": "choice", "choice": choice, "probabilities": distribution,
                "confidence": top}  # fmt: skip
    last = len(question.criteria) - 1.0
    return {"type": "score", "score": min(max(expected_level(distribution), 0.0), last),
            "legend": {str(k): text for k, text in enumerate(question.criteria)},
            "probabilities": distribution, "confidence": top}  # fmt: skip


def installed_version(package: str) -> str | None:
    """The installed distribution's version, with the commit when it came from a repository."""
    from importlib import metadata

    try:
        dist = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return None
    version = dist.version
    direct = dist.read_text("direct_url.json")
    if direct:
        commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
        if commit:
            version += f" at commit {commit}"
    return version


def runtime_versions() -> str:
    """torch and transformers as installed, for the rows' path_used."""
    parts = []
    for name in ("torch", "transformers"):
        version = installed_version(name)
        if version:
            parts.append(f"{name} {version}")
    return ", ".join(parts)
