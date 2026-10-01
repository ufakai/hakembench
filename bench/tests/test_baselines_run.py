"""The step 9 runner: where rows may go, unanswerable items, resume and the run log. No model."""

import json
import time

import pytest

from bench.adapters.base import AdapterResult, ModelInfo
from bench.baselines.run import REPO_ROOT, check_destination, meta_paths, run_model
from bench.harness.items import Item
from schema.api import Response

ITEMS = [
    Item.model_validate({
        "id": name, "track": "spam", "state": f"metin {name}",
        "questions": {"spam": {"type": "noul", "instructions": "Spam mı?"},
                      "tür": {"type": "choice", "instructions": "Hangi tür?",
                              "criteria": {"reklam": None, "kişisel": None}}},
        "gold": {"spam": True, "tür": "reklam"},
    })
    for name in ("a", "b", "c")
]  # fmt: skip


class FakeAdapter:
    """Refuses item b by its tokenisation; answers the others."""

    def __init__(self):
        self.answered = []

    def info(self):
        return ModelInfo(adapter="fake", model="org/fake", revision="abc123",
                         path_used="fake route", device="cpu")  # fmt: skip

    def unanswerable(self, request):
        return "too long" if request.state == "metin b" else None

    def answer(self, request):
        self.answered.append(request.state)
        started = time.perf_counter()
        answers = {"spam": {"type": "noul", "noul": 0.9},
                   "tür": {"type": "choice", "choice": "reklam", "confidence": 0.6,
                           "probabilities": {"reklam": 0.6, "kişisel": 0.4}}}  # fmt: skip
        return AdapterResult(
            response=Response.model_validate({"model": "org/fake", "answers": answers}),
            latency_ms=(time.perf_counter() - started) * 1000,
            confidence_source={"tür": "max_probability"},
        )


def go(out, adapter=None, resume=False, limit=0):
    return run_model(ITEMS, adapter or FakeAdapter(), out, items_path=out.parent / "items.jsonl",
                     resume=resume, limit=limit, git_commit="c0ffee", threads=8)  # fmt: skip


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_answerable_items_get_rows_and_the_rest_are_recorded_and_counted(tmp_path):
    out = tmp_path / "runs" / "fake-public.jsonl"
    adapter = FakeAdapter()
    record = go(out, adapter)
    assert adapter.answered == ["metin a", "metin c"]
    written = rows(out)
    assert {(r["item_id"], r["question_id"]) for r in written} == {
        ("a", "spam"), ("a", "tür"), ("c", "spam"), ("c", "tür")}  # fmt: skip
    assert {r["script"] for r in written} == {"bench/baselines/run.py"}
    assert {r["model_revision"] for r in written} == {"abc123"}

    unanswered_path, log_path = meta_paths(out)
    (refused,) = rows(unanswered_path)
    assert refused["item_id"] == "b"
    assert refused["reason"] == "too long"
    assert refused["question_ids"] == ["spam", "tür"]
    (logged,) = rows(log_path)
    assert logged == record
    assert (record["items_run"], record["items_unanswerable"], record["questions_unanswerable"],
            record["rows_written"], record["threads"]) == (2, 1, 2, 4, 8)  # fmt: skip
    assert record["wall_seconds"] >= 0


def test_resume_skips_answered_and_already_refused_items(tmp_path):
    out = tmp_path / "runs" / "fake-public.jsonl"
    go(out, limit=1)
    assert [r["item_id"] for r in rows(out)] == ["a", "a"]
    adapter = FakeAdapter()
    second = go(out, adapter, resume=True)
    assert adapter.answered == ["metin c"]
    assert (second["items_already_answered"], second["items_unanswerable"]) == (1, 1)
    third = go(out, FakeAdapter(), resume=True)
    assert (third["items_run"], third["items_unanswerable_before"], third["rows_written"]) == (
        0, 1, 0)  # fmt: skip
    assert len(rows(meta_paths(out)[0])) == 1  # b is recorded once


def test_an_existing_rows_file_needs_resume(tmp_path):
    out = tmp_path / "fake.jsonl"
    go(out)
    with pytest.raises(FileExistsError):
        go(out)


@pytest.mark.parametrize(
    ("model", "items", "out", "ok"),
    [
        ("laya", "bench/hakembench/v1.0/public.jsonl",
         "results/step9/runs/laya-public.jsonl", True),
        ("laya", "results/private/step8/v1.0/private.jsonl",
         "results/private/step9/runs/laya-private.jsonl", True),
        ("laya", "results/private/step8/v1.0/private.jsonl",
         "results/step9/runs/laya-private.jsonl", False),
        ("laya", "bench/hakembench/v1.0/public.jsonl", "results/step8/runs/laya.jsonl", False),
    ],
)  # fmt: skip
def test_rows_only_go_where_they_may(model, items, out, ok, monkeypatch):
    if ok:
        check_destination(model, REPO_ROOT / items, REPO_ROOT / out)
    else:
        with pytest.raises(ValueError):
            check_destination(model, REPO_ROOT / items, REPO_ROOT / out)
