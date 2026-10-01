"""The smoke command: rows on success, exit 1 and no overwrite otherwise."""

import json

import pytest

from bench.adapters.base import AdapterError
from bench.harness import cli
from bench.harness.results import DirtyTreeError
from bench.tests.test_harness import ANSWERS, SMOKE, FakeAdapter


@pytest.fixture
def fake_route(monkeypatch):
    monkeypatch.setattr(cli, "_adapter", lambda args: FakeAdapter(ANSWERS))
    monkeypatch.setattr(cli, "committed_code_version", lambda repo_root: "c0ffee")


def test_smoke_writes_one_row_per_question(tmp_path, fake_route, capsys):
    out = tmp_path / "step0/smoke.jsonl"
    assert cli.main(["smoke", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["question_type"] for row in rows] == ["choice", "noul", "score"]
    assert {row["script"] for row in rows} == {"bench/harness/cli.py smoke"}
    assert "3 rows" in capsys.readouterr().out


def test_limit_runs_only_the_first_items(tmp_path, fake_route):
    out = tmp_path / "smoke.jsonl"
    assert cli.main(["smoke", "--out", str(out), "--limit", "1"]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_an_existing_results_file_is_not_overwritten(tmp_path, fake_route, capsys):
    out = tmp_path / "smoke.jsonl"
    out.write_text("earlier measurement\n", encoding="utf-8")
    assert cli.main(["smoke", "--out", str(out)]) == 1
    assert out.read_text(encoding="utf-8") == "earlier measurement\n"
    assert "never regenerated silently" in capsys.readouterr().err

    assert cli.main(["smoke", "--out", str(out), "--append"]) == 0
    assert out.read_text(encoding="utf-8").startswith("earlier measurement\n")
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4


def test_a_dirty_tree_stops_the_run(tmp_path, monkeypatch, capsys):
    def dirty(repo_root):
        raise DirtyTreeError("uncommitted changes in the working tree")

    monkeypatch.setattr(cli, "_adapter", lambda args: FakeAdapter(ANSWERS))
    monkeypatch.setattr(cli, "committed_code_version", dirty)
    out = tmp_path / "smoke.jsonl"
    assert cli.main(["smoke", "--out", str(out)]) == 1
    assert "uncommitted" in capsys.readouterr().err
    assert not out.exists()


def test_an_adapter_failure_is_reported_without_a_traceback(tmp_path, monkeypatch, capsys):
    def broken(args):
        raise AdapterError("server did not return log-probabilities")

    monkeypatch.setattr(cli, "committed_code_version", lambda repo_root: "c0ffee")
    monkeypatch.setattr(cli, "_adapter", broken)
    out = tmp_path / "smoke.jsonl"
    assert cli.main(["smoke", "--out", str(out)]) == 1
    assert "log-probabilities" in capsys.readouterr().err
    assert not out.exists()


def test_the_server_route_needs_a_model_and_a_url(monkeypatch):
    monkeypatch.setattr(cli, "committed_code_version", lambda repo_root: "c0ffee")
    with pytest.raises(SystemExit):
        cli.main(["smoke", "--adapter", "chat-completions"])


def items_file(tmp_path):
    """Two items: one asking two questions, one asking one."""
    smoke = [json.loads(x) for x in SMOKE.read_text(encoding="utf-8").splitlines() if x.strip()]
    first = {**smoke[0], "id": "a", "questions": {**smoke[0]["questions"], **smoke[1]["questions"]},
             "gold": {**smoke[0]["gold"], **smoke[1]["gold"]}}  # fmt: skip
    path = tmp_path / "items.jsonl"
    lines = [first, {**smoke[2], "id": "b"}]
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), "utf-8")
    return path


def rows_of(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_writes_one_row_per_question(tmp_path, fake_route, capsys):
    out = tmp_path / "runs/fake.jsonl"
    args = ["run", "--adapter", "local", "--items", str(items_file(tmp_path)), "--out", str(out)]
    assert cli.main(args) == 0
    rows = rows_of(out)
    assert [(r["item_id"], r["question_id"]) for r in rows] == [
        ("a", "birim"), ("a", "acil"), ("b", "memnuniyetsizlik")
    ]  # fmt: skip
    assert {r["script"] for r in rows} == {"bench/harness/cli.py run"}
    assert rows[1]["gold"] is True
    assert "3 rows" in capsys.readouterr().out


def test_resume_skips_the_items_already_answered(tmp_path, fake_route, capsys):
    items, out = items_file(tmp_path), tmp_path / "fake.jsonl"
    base = ["run", "--adapter", "local", "--items", str(items), "--out", str(out)]
    assert cli.main([*base, "--limit", "1"]) == 0
    assert {r["item_id"] for r in rows_of(out)} == {"a"}
    assert cli.main([*base, "--resume"]) == 0
    assert [r["item_id"] for r in rows_of(out)] == ["a", "a", "b"]
    assert "1 items already answered" in capsys.readouterr().out
    assert cli.main([*base, "--resume"]) == 0
    assert len(rows_of(out)) == 3


def test_resume_counts_only_rows_of_the_same_model(tmp_path, fake_route):
    items, out = items_file(tmp_path), tmp_path / "fake.jsonl"
    base = ["run", "--adapter", "local", "--items", str(items), "--out", str(out)]
    assert cli.main([*base, "--limit", "1"]) == 0
    other = [{**r, "model_revision": "rev0"} for r in rows_of(out)]
    out.write_text("".join(json.dumps(r) + "\n" for r in other), encoding="utf-8")
    assert cli.main([*base, "--resume"]) == 0
    assert [r["item_id"] for r in rows_of(out)[2:]] == ["a", "a", "b"]


def test_resume_counts_only_rows_of_the_same_prompt(tmp_path, fake_route):
    items, out = items_file(tmp_path), tmp_path / "fake.jsonl"
    base = ["run", "--adapter", "local", "--items", str(items), "--out", str(out)]
    assert cli.main([*base, "--limit", "1"]) == 0
    other = [{**r, "prompt_version": "labels-v1+en"} for r in rows_of(out)]
    out.write_text("".join(json.dumps(r) + "\n" for r in other), encoding="utf-8")
    assert cli.main([*base, "--resume"]) == 0
    assert [r["item_id"] for r in rows_of(out)[2:]] == ["a", "a", "b"]


def test_an_item_answered_in_part_is_refused(tmp_path, fake_route, capsys):
    items, out = items_file(tmp_path), tmp_path / "fake.jsonl"
    base = ["run", "--adapter", "local", "--items", str(items), "--out", str(out)]
    assert cli.main([*base, "--limit", "1"]) == 0
    out.write_text(out.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    assert cli.main([*base, "--resume"]) == 1
    assert "item a has rows" in capsys.readouterr().err


def test_run_refuses_an_existing_out_without_resume(tmp_path, fake_route, capsys):
    out = tmp_path / "fake.jsonl"
    out.write_text("earlier measurement\n", encoding="utf-8")
    args = ["run", "--adapter", "local", "--items", str(items_file(tmp_path)), "--out", str(out)]
    assert cli.main(args) == 1
    assert "--resume" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "earlier measurement\n"


KARAR = pytest.mark.skipif(
    "karar" not in cli.routes(),
    reason="the karar adapter is in the ufakzeka-karar code repository only",
)


@KARAR
def test_the_karar_route_needs_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "committed_code_version", lambda repo_root: "c0ffee")
    args = ["run", "--adapter", "karar", "--items", str(items_file(tmp_path)),
            "--out", str(tmp_path / "x.jsonl")]  # fmt: skip
    with pytest.raises(SystemExit, match="--weights"):
        cli.main(args)


def test_prompt_lang_reaches_the_chat_route_and_is_refused_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "committed_code_version", lambda repo_root: "c0ffee")
    items = str(items_file(tmp_path))
    for adapter in [r for r in ("karar", "local") if r in cli.routes()]:
        args = ["run", "--adapter", adapter, "--items", items, "--weights", "w.pt",
                "--out", str(tmp_path / f"{adapter}.jsonl"), "--prompt-lang", "en"]  # fmt: skip
        with pytest.raises(SystemExit, match="chat-completions route only"):
            cli.main(args)
    made = []

    class Recorder:
        def __init__(self, **kwargs):
            made.append(kwargs)
            raise AdapterError("no server in tests")

    import bench.adapters.chat_completions as chat

    monkeypatch.setattr(chat, "ChatCompletionsAdapter", Recorder)
    base = ["run", "--adapter", "chat-completions", "--items", items, "--model", "m",
            "--base-url", "http://localhost:1/v1"]  # fmt: skip
    assert cli.main([*base, "--out", str(tmp_path / "en.jsonl"), "--prompt-lang", "en"]) == 1
    assert cli.main([*base, "--out", str(tmp_path / "tr.jsonl")]) == 1
    assert [m["prompt_lang"] for m in made] == ["en", "tr"]


def test_the_run_parser_offers_only_routes_whose_adapters_are_present(tmp_path, capsys):
    adapters = tmp_path / "bench/adapters"
    adapters.mkdir(parents=True)
    for name in ("local", "chat_completions", "stated", "typesafe"):
        (adapters / f"{name}.py").write_text("")
    # A copy without the karar adapter, as in HakemBench's own repository.
    assert cli.routes(tmp_path) == [
        "local",
        "chat-completions",
        "typesafe",
        "hosted-labels",
        "hosted-stated",
    ]
    (adapters / "typesafe.py").unlink()
    assert cli.routes(tmp_path) == ["local", "chat-completions", "hosted-labels", "hosted-stated"]
    (adapters / "stated.py").unlink()
    assert cli.routes(tmp_path) == ["local", "chat-completions", "hosted-labels"]
    if "karar" not in cli.routes():
        with pytest.raises(SystemExit):
            cli.main(["run", "--adapter", "karar", "--items", "x", "--out", "y"])
        assert "invalid choice: 'karar'" in capsys.readouterr().err


def test_a_route_whose_packages_are_missing_stops_with_one_line(tmp_path, monkeypatch, capsys):
    from bench.harness import cli

    def no_torch(args):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(cli, "_adapter", no_torch)
    monkeypatch.setattr(cli, "committed_code_version", lambda root: "0" * 40)
    code = cli.main(["smoke", "--out", str(tmp_path / "rows.jsonl")])
    assert code == 1
    err = capsys.readouterr().err
    assert "needs the package 'torch'" in err and "Traceback" not in err
