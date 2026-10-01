"""Typed questions and typed answers.

The shapes follow the public typed-decision contract as documented at
docs.typesafe.ai on 2026-09-19 (pages api, primitives/choice,
primitives/score, primitives/noul, confidence), so a client written for
that contract can talk to our server unchanged.

Two numbers that are not a contradiction
----------------------------------------
A choice question accepts up to MAX_CHOICE_OPTIONS (255) options. That is
the API limit, taken from the public contract. The ufakzeka-karar head
scores at most HEAD_OPTIONS_PER_PASS (10) options in one forward pass
(docs/PLAN.md, "Head and training"). A larger option set is split into
chunks of at most 10 inside the server, each chunk is scored, and the chunk
winners are scored again against each other. Callers never see the chunking
and never need to stay under 10. The first number belongs to the API, the
second to the model, and both are correct.

Our one extension to the contract is `abstain`: the probability that the
model should answer "emin değilim" on this question. It is an added,
optional field, so responses from models without an abstain output (every
external baseline) leave it as None and still validate.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# API limit on options in one choice question (public contract).
MAX_CHOICE_OPTIONS = 255
# The contract states no minimum. One option is not a decision, so we ask for two.
MIN_CHOICE_OPTIONS = 2
# Score levels (public contract: at least two, up to 10).
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
# Model limit, not an API limit. See the module docstring.
HEAD_OPTIONS_PER_PASS = 10

# Probabilities arrive as JSON floats, often rounded to two or three places.
PROBABILITY_SUM_TOLERANCE = 1e-3

# State and instructions are a string or any JSON object or array.
JsonContent = str | dict[str, Any] | list[Any]

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulCriteria(_Strict):
    """Optional descriptions of what a yes and a no mean."""

    true: str | None = None
    false: str | None = None


class NoulQuestion(_Strict):
    """A yes/no question. The answer is the probability of yes."""

    type: Literal["noul"] = "noul"
    instructions: JsonContent
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Strict):
    """One option out of a set. `criteria` maps each option to a description, or to None."""

    type: Literal["choice"] = "choice"
    instructions: JsonContent
    criteria: dict[str, str | None]

    @model_validator(mode="after")
    def _check_options(self) -> ChoiceQuestion:
        n = len(self.criteria)
        if not MIN_CHOICE_OPTIONS <= n <= MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"a choice question takes {MIN_CHOICE_OPTIONS} to {MAX_CHOICE_OPTIONS} "
                f"options, got {n}"
            )
        if any(not option.strip() for option in self.criteria):
            raise ValueError("option names must not be empty")
        return self

    @property
    def options(self) -> list[str]:
        return list(self.criteria)


class ScoreQuestion(_Strict):
    """A position on an ordered rubric. Level k is the k-th entry of `criteria`, from 0."""

    type: Literal["score"] = "score"
    instructions: JsonContent
    criteria: list[str]

    @model_validator(mode="after")
    def _check_levels(self) -> ScoreQuestion:
        n = len(self.criteria)
        if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"a score question takes {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {n}"
            )
        if any(not level.strip() for level in self.criteria):
            raise ValueError("level descriptions must not be empty")
        return self


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


def _check_distribution(probabilities: dict[str, float]) -> None:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    total = math.fsum(probabilities.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"probabilities sum to {total!r}, expected 1")


class NoulAnswer(_Strict):
    """The contract gives a noul answer one number and no confidence."""

    type: Literal["noul"] = "noul"
    noul: Probability
    abstain: Probability | None = None


class ChoiceAnswer(_Strict):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability
    abstain: Probability | None = None

    @model_validator(mode="after")
    def _check(self) -> ChoiceAnswer:
        _check_distribution(self.probabilities)
        if self.choice not in self.probabilities:
            raise ValueError(f"choice {self.choice!r} is not one of the options")
        # Ties are allowed; the chosen option must be one of the most probable.
        if self.probabilities[self.choice] < max(self.probabilities.values()):
            raise ValueError("choice is not a highest-probability option")
        return self


class ScoreAnswer(_Strict):
    """`score` is the probability-weighted level and can land between levels."""

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, Probability]
    confidence: Probability
    abstain: Probability | None = None

    @model_validator(mode="after")
    def _check(self) -> ScoreAnswer:
        _check_distribution(self.probabilities)
        expected_keys = [str(k) for k in range(len(self.legend))]
        if sorted(self.legend, key=int) != expected_keys:
            raise ValueError("legend keys must be the level indices 0..K-1 as strings")
        if set(self.probabilities) != set(self.legend):
            raise ValueError("probabilities and legend must cover the same levels")
        if not 0.0 <= self.score <= len(self.legend) - 1:
            raise ValueError("score lies outside the level range")
        return self

    @property
    def argmax_level(self) -> int:
        return max(range(len(self.legend)), key=lambda k: self.probabilities[str(k)])


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


def expected_level(probabilities: dict[str, float]) -> float:
    """The probability-weighted level, the contract's definition of `score`."""
    return math.fsum(int(level) * p for level, p in probabilities.items())
