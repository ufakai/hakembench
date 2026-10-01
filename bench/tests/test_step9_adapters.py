"""The step 9 adapters (decider, Kev, simple-jev) with tiny fake models: no torch, no network.

Each fake answers in the shape its project's code returns (decider's
systemone.format_answer, Kev's api.to_answers, simple-jev's
common.response_scoring), read at the project commits the result rows record
in path_used.
"""

import asyncio
import json

import pytest

from bench.adapters.base import AdapterError
from bench.adapters.decider import DeciderAdapter
from bench.adapters.kev import KevAdapter
from bench.adapters.simple_jev import SimpleJevAdapter
from bench.adapters.systemone import Unanswerable, body, to_answer
from bench.harness.items import Item
from bench.harness.results import ResultRow, ResultsWriter
from bench.harness.runner import run
from schema.api import Request

ITEM = Item.model_validate(
    {
        "id": "t-1",
        "track": "yonlendirme",
        "state": "Kartımdan iki kez ödeme alındı, iade istiyorum.",
        "questions": {
            "iade": {"type": "noul", "instructions": "Müşteri iade istiyor mu?"},
            "birim": {
                "type": "choice",
                "instructions": "Bu talebi hangi birim ele almalı?",
                "criteria": {"fatura": "Ödeme ve iade", "teknik": None, "satis": None},
            },
            "ofke": {
                "type": "score",
                "instructions": "Müşteri ne kadar öfkeli?",
                "criteria": ["Sakin", "Kızgın", "Çok öfkeli"],
            },
        },
        "gold": {"iade": True, "birim": "fatura", "ofke": 1},
    }
)
REQUEST = Request(state=ITEM.state, model="m", questions=ITEM.questions)
RUNTIME = {"gpu": "fake", "torch": "0"}


def returned(wire: dict, extra: bool = False) -> dict:
    """What the three projects return for REQUEST: the same numbers, four places."""
    assert wire["state"] == ITEM.state
    answers = {
        "iade": {"type": "noul", "noul": 0.9731},
        "birim": {"type": "choice", "choice": "fatura", "confidence": 0.7,
                  "probabilities": {"fatura": 0.8, "teknik": 0.15, "satis": 0.05}},
        "ofke": {"type": "score", "score": 1.1, "confidence": 0.55,
                 "legend": {"0": "Sakin", "1": "Kızgın", "2": "Çok öfkeli"},
                 "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}},
    }  # fmt: skip
    if extra:
        answers["birim"] |= {"x_p_max": 0.8, "certainty": 0.4}
        answers["ofke"] |= {"level_fit": {"0": 0.2, "1": 0.5, "2": 0.3}, "fit_mass": 1.0}
    return {"model": "x", "answers": answers, "usage": {"input_tokens": 41, "output_tokens": 0}}


class FakeDecider:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.seen: list = []

    def system_one(self, state, questions):
        self.seen.append((state, questions))
        if self.error:
            raise self.error
        return returned({"state": state, "questions": questions}, extra=True)


class FakeService:
    metadata = {"prompt_policy": "shared_examples_binary"}

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.seen: list = []

    async def classify(self, request):
        await asyncio.sleep(0)
        self.seen.append(request)
        if self.error:
            raise self.error
        return returned(request)


class Http422(Exception):
    status_code = 422


def decider(fake):
    return DeciderAdapter(fake, model_id="Mapika/decider-2b", revision="abc", runtime=RUNTIME)


def kev(call):
    return KevAdapter(call, model_id="jaredpalmer/kev-4b", revision="def", runtime=RUNTIME)


def simple_jev(service):
    return SimpleJevAdapter(service, model_id="Qwen/Qwen3.5-4B", revision="123", runtime=RUNTIME)


def test_the_body_is_the_item_as_it_stands():
    wire = body(REQUEST)
    assert wire["state"] == ITEM.state
    assert set(wire) == {"state", "questions"}
    # A noul without criteria leaves the field out; an option without a description stays null.
    assert wire["questions"]["iade"] == {"type": "noul", "instructions": "Müşteri iade istiyor mu?"}
    assert wire["questions"]["birim"]["criteria"] == {"fatura": "Ödeme ve iade", "teknik": None,
                                                      "satis": None}  # fmt: skip
    assert wire["questions"]["ofke"]["criteria"] == ["Sakin", "Kızgın", "Çok öfkeli"]
    assert list(wire["questions"]) == list(ITEM.questions)


@pytest.mark.parametrize("make", ["decider", "kev", "simple_jev"])
def test_every_adapter_returns_the_models_numbers_unchanged(make):
    adapter = {
        "decider": lambda: decider(FakeDecider()),
        "kev": lambda: kev(lambda wire: returned(wire)),
        "simple_jev": lambda: simple_jev(FakeService()),
    }[make]()
    result = adapter.answer(REQUEST)
    answers = result.response.answers
    assert answers["iade"].noul == 0.9731
    assert answers["birim"].choice == "fatura"
    assert answers["birim"].probabilities == {"fatura": 0.8, "teknik": 0.15, "satis": 0.05}
    assert answers["birim"].confidence == 0.7
    assert answers["ofke"].score == 1.1
    assert answers["ofke"].probabilities == {"0": 0.2, "1": 0.5, "2": 0.3}
    assert answers["ofke"].legend == {"0": "Sakin", "1": "Kızgın", "2": "Çok öfkeli"}
    # simple-jev's confidence is its plain top probability; the others return their own.
    own = "max_probability" if make == "simple_jev" else "model"
    assert result.confidence_source == {"birim": own, "ofke": own}
    assert result.response.usage.input_tokens == 41
    assert all(a.abstain is None for a in answers.values())


def test_decider_extra_fields_become_diagnostics():
    result = decider(FakeDecider()).answer(REQUEST)
    assert result.diagnostics["birim"] == {"x_p_max": 0.8, "certainty": 0.4}
    assert result.diagnostics["ofke"]["fit_mass"] == 1.0


def test_simple_jev_gets_the_served_model_name_and_nothing_else():
    service = FakeService()
    simple_jev(service).answer(REQUEST)
    assert service.seen == [{"model": "Qwen/Qwen3.5-4B", **body(REQUEST)}]
    assert simple_jev(service).info().prompt_version == "simple-jev shared_examples_binary"


@pytest.mark.parametrize(
    "adapter",
    [
        lambda: decider(FakeDecider(ValueError("choice criteria: a map of 2..255 options"))),
        lambda: kev(lambda wire: (_ for _ in ()).throw(Http422("state too long"))),
        lambda: simple_jev(FakeService(ValueError("prompt exceeds 16384 tokens"))),
    ],
)
def test_a_limit_the_model_enforces_is_unanswerable(adapter):
    with pytest.raises(Unanswerable):
        adapter().answer(REQUEST)


def test_other_failures_are_not_counted_as_unanswered():
    def broken(wire):
        raise RuntimeError("CUDA error")

    with pytest.raises(RuntimeError):
        kev(broken).answer(REQUEST)


def test_an_answer_that_misses_an_option_stops_the_run():
    def partial(wire):
        body_ = returned(wire)
        del body_["answers"]["birim"]["probabilities"]["satis"]
        return body_

    with pytest.raises(AdapterError, match="cover the options"):
        kev(partial).answer(REQUEST)


def test_a_missing_answer_stops_the_run():
    def missing(wire):
        body_ = returned(wire)
        del body_["answers"]["ofke"]
        return body_

    with pytest.raises(AdapterError, match="no answer for ofke"):
        kev(missing).answer(REQUEST)


def test_a_score_past_the_top_by_rounding_is_clipped_and_beyond_is_refused():
    question = ITEM.questions["ofke"]
    raw = returned({"state": ITEM.state})["answers"]["ofke"]
    answer, _ = to_answer(question, raw | {"score": 2.0000000001})
    assert answer["score"] == 2.0
    with pytest.raises(AdapterError, match="outside"):
        to_answer(question, raw | {"score": 2.1})


def test_rows_through_the_harness_record_route_versions_and_revision(tmp_path):
    out = tmp_path / "rows.jsonl"
    with ResultsWriter(out) as writer:
        run([ITEM], decider(FakeDecider()), writer, repo_root=tmp_path, script="t", git_commit="c0")
    rows = [ResultRow.model_validate(json.loads(line)) for line in out.read_text().splitlines()]
    assert [r.question_id for r in rows] == ["iade", "birim", "ofke"]
    assert {r.adapter for r in rows} == {"decider"}
    assert {r.model_revision for r in rows} == {"abc"}
    assert all("gpu fake" in r.path_used and "torch 0" in r.path_used for r in rows)
    assert rows[1].confidence_source == "model" and rows[0].confidence_source is None
    assert rows[1].gold == "fatura"
