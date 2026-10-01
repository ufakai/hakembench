"""Items, the append-only writer, the clean-tree rule and the runner."""

import json
import subprocess
import time
from pathlib import Path

import pytest

from bench.adapters.base import AdapterError, AdapterResult, ModelInfo
from bench.harness.items import Item, load_items
from bench.harness.results import (
    DirtyTreeError,
    ResultRow,
    ResultsWriter,
    committed_code_version,
)
from bench.harness.runner import run
from schema.api import Response

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = REPO_ROOT / "bench/items/smoke.jsonl"


def test_the_smoke_items_load_and_cover_every_question_type():
    items = load_items(SMOKE)
    types = {q.type for item in items for q in item.questions.values()}
    assert types == {"choice", "score", "noul"}
    assert all(item.gold for item in items)


@pytest.mark.parametrize(
    "gold",
    [{"q": "yok"}, {"q": 1}, {"başka": "a"}],
)
def test_gold_must_fit_the_question(gold):
    body = {
        "id": "x",
        "track": "t",
        "state": "s",
        "questions": {
            "q": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}}
        },
        "gold": gold,
    }
    with pytest.raises(ValueError):
        Item.model_validate(body)


def test_a_bool_is_not_a_score_level_and_an_int_is_not_a_noul_label():
    score = {"type": "score", "instructions": "?", "criteria": ["az", "çok"]}
    noul = {"type": "noul", "instructions": "?"}
    base = {"id": "x", "track": "t", "state": "s"}
    with pytest.raises(ValueError):
        Item.model_validate({**base, "questions": {"q": score}, "gold": {"q": True}})
    with pytest.raises(ValueError):
        Item.model_validate({**base, "questions": {"q": noul}, "gold": {"q": 1}})


def test_duplicate_item_ids_are_refused(tmp_path):
    line = SMOKE.read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "items.jsonl"
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_items(path)


def _row(**changes) -> ResultRow:
    fields = dict(
        run_id="r",
        created_utc="2026-09-19T00:00:00+00:00",
        git_commit="abc",
        script="test",
        adapter="fake",
        model="m",
        model_revision=None,
        path_used="test double",
        device=None,
        prompt_version=None,
        item_id="i",
        track="t",
        question_id="q",
        question_type="noul",
        answer={"type": "noul", "noul": 0.5},
        confidence_source=None,
        gold=True,
        latency_ms=1.0,
        latency_scope="question",
        host="test",
    )
    return ResultRow(**{**fields, **changes})


def test_the_writer_never_overwrites(tmp_path):
    path = tmp_path / "results/step0/rows.jsonl"
    with ResultsWriter(path) as writer:
        writer.write(_row(item_id="first"))
    with pytest.raises(FileExistsError, match="never regenerated silently"):
        ResultsWriter(path)
    with ResultsWriter(path, append=True) as writer:
        writer.write(_row(item_id="second"))
    ids = [json.loads(line)["item_id"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert ids == ["first", "second"]


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def tiny_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "first")
    return tmp_path


def test_a_clean_tree_gives_the_commit(tiny_repo):
    assert len(committed_code_version(tiny_repo)) == 40


def test_uncommitted_code_stops_the_run(tiny_repo):
    (tiny_repo / "code.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(DirtyTreeError):
        committed_code_version(tiny_repo)


def test_an_untracked_script_stops_the_run(tiny_repo):
    (tiny_repo / "new_script.py").write_text("", encoding="utf-8")
    with pytest.raises(DirtyTreeError):
        committed_code_version(tiny_repo)


def test_new_rows_under_results_do_not_count_as_dirty(tiny_repo):
    (tiny_repo / "results").mkdir()
    (tiny_repo / "results/rows.jsonl").write_text("{}\n", encoding="utf-8")
    committed_code_version(tiny_repo)


class FakeAdapter:
    """Answers with fixed values and records what it was asked."""

    def __init__(self, answers, per_question=False):
        self.answers = answers
        self.per_question = per_question
        self.requests = []

    def info(self):
        return ModelInfo(
            adapter="fake", model="fake-1", revision="rev1", path_used="test double", device="cpu"
        )

    def answer(self, request):
        self.requests.append(request)
        time.sleep(0)
        answers = {qid: self.answers[qid] for qid in request.questions}
        response = Response.model_validate(
            {
                "model": "fake-1",
                "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        )
        return AdapterResult(
            response=response,
            latency_ms=12.5,
            per_question_ms={qid: 4.0 for qid in answers} if self.per_question else None,
            confidence_source={qid: "max_probability" for qid in answers},
            diagnostics={qid: {"label_mass": 0.9} for qid in answers},
        )


ANSWERS = {
    "birim": {
        "type": "choice",
        "choice": "teknik destek",
        "probabilities": {"teknik destek": 0.8, "faturalandırma": 0.1, "satış": 0.1},
        "confidence": 0.8,
    },
    "acil": {"type": "noul", "noul": 0.9},
    "memnuniyetsizlik": {
        "type": "score",
        "score": 2.5,
        "legend": {
            "0": "Memnun ya da nötr",
            "1": "Biraz rahatsız",
            "2": "Belirgin biçimde memnuniyetsiz",
            "3": "Çok öfkeli, müşteriyi kaybetme riski var",
        },
        "probabilities": {"0": 0.0, "1": 0.1, "2": 0.3, "3": 0.6},
        "confidence": 0.6,
    },
}


def test_the_runner_writes_one_self_describing_row_per_question(tmp_path):
    items = load_items(SMOKE)
    adapter = FakeAdapter(ANSWERS, per_question=True)
    path = tmp_path / "rows.jsonl"
    with ResultsWriter(path) as writer:
        run_id = run(items, adapter, writer, repo_root=tmp_path, script="test", git_commit="c0ffee")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 3
    first = rows[0]
    assert first["run_id"] == run_id and first["git_commit"] == "c0ffee"
    assert (first["adapter"], first["model"], first["model_revision"]) == ("fake", "fake-1", "rev1")
    assert first["path_used"] == "test double"
    assert (first["item_id"], first["question_id"], first["question_type"]) == (
        "smoke-001",
        "birim",
        "choice",
    )
    assert first["answer"]["choice"] == "teknik destek" and first["gold"] == "teknik destek"
    assert first["confidence_source"] == "max_probability"
    assert first["latency_ms"] == 4.0 and first["latency_scope"] == "question"
    assert first["diagnostics"] == {"label_mass": 0.9}
    assert rows[1]["gold"] is True and rows[2]["gold"] == 3
    # The request carries the adapter's model id and the item's state unchanged.
    assert adapter.requests[0].model == "fake-1"
    assert adapter.requests[0].state == items[0].state


def test_a_malformed_answer_stops_the_run_and_writes_nothing(tmp_path):
    wrong = {**ANSWERS, "birim": ANSWERS["acil"]}
    path = tmp_path / "rows.jsonl"
    with ResultsWriter(path) as writer, pytest.raises(ValueError, match="asked a choice"):
        run(
            load_items(SMOKE),
            FakeAdapter(wrong),
            writer,
            repo_root=tmp_path,
            script="test",
            git_commit="c0ffee",
        )
    # Nothing was answered validly, so no file was created at all.
    assert not path.exists()


def test_adapter_error_is_a_runtime_error():
    assert issubclass(AdapterError, RuntimeError)


def test_option_order_sets_the_shown_order_and_must_be_an_order_of_the_options():
    from bench.harness.items import Item

    row = {
        "id": "x",
        "track": "t",
        "state": "metin",
        "questions": {"q": {"type": "choice", "instructions": "?",
                            "criteria": {"a": None, "b": "B", "c": None}}},
        "gold": {"q": "b"},
    }  # fmt: skip
    # A reader that sorted the map's keys: the list restores the order shown.
    item = Item.model_validate({**row, "option_order": {"q": ["c", "a", "b"]}})
    assert list(item.questions["q"].criteria) == ["c", "a", "b"]
    assert item.questions["q"].criteria["b"] == "B"
    assert list(Item.model_validate(row).questions["q"].criteria) == ["a", "b", "c"]
    for bad in ({"q": ["a", "b"]}, {"q": ["a", "b", "b"]}, {"z": ["a"]}):
        with pytest.raises(ValueError, match="option_order"):
            Item.model_validate({**row, "option_order": bad})
