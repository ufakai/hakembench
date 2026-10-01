"""The board on synthetic results rows whose answers are known."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("relplot")
pytest.importorskip("scipy")

from bench import board  # noqa: E402
from bench.harness.results import ResultRow  # noqa: E402

CONFIG = {"axes": {name: 1.0 for name in board.AXES}, "speed_reference_ms": 100.0,
          "cost_reference_usd_per_1000": 1.0, "safety_positive": {"spam": "true"}}  # fmt: skip
DRAWS = 25


def row(model, item, track, qid, kind, answer, gold, **extra):
    base = {"run_id": "r", "created_utc": "2026-09-25T00:00:00+00:00", "git_commit": "abc",
            "script": "test", "adapter": "local", "model": model, "model_revision": "v1",
            "path_used": "test", "device": "cpu", "prompt_version": None, "item_id": item,
            "track": track, "question_id": qid, "question_type": kind, "answer": answer,
            "confidence_source": "max_probability" if kind != "noul" else None, "gold": gold,
            "latency_ms": 50.0, "latency_scope": "question", "host": "test"}  # fmt: skip
    return ResultRow.model_validate({**base, **extra})


def choice(p_right, right, other):
    probs = {right: p_right, other: 1 - p_right}
    top = max(probs, key=probs.get)
    return {"type": "choice", "choice": top, "probabilities": probs,
            "confidence": max(probs.values())}  # fmt: skip


def rows_for(model, skill, n=40, tracks=("destek", "spam"), public_skill=None, seed=0, **extra):
    """One choice question per item, plus a noul on spam items; skill is the chance of a hit."""
    rng = np.random.default_rng(seed)
    out = []
    for track in tracks:
        for i in range(n):
            item = f"{track}-{i:03d}"
            s = public_skill if public_skill is not None and i % 2 == 0 else skill
            gold, wrong = ("a", "b") if i % 3 else ("b", "a")
            p = rng.uniform(0.55, 0.95)
            p_right = p if rng.uniform() < s else 1 - p
            out.append(row(model, item, track, "q", "choice", choice(p_right, gold, wrong), gold,
                           **extra))  # fmt: skip
            if track == "spam":
                truth = bool(i % 2)
                yes = p if (rng.uniform() < s) == truth else 1 - p
                out.append(row(model, item, track, "is_spam", "noul",
                               {"type": "noul", "noul": yes}, truth, **extra))  # fmt: skip
    return out


def splits_for(n=40, tracks=("destek", "spam")):
    return {f"{t}-{i:03d}": "public" if i % 2 == 0 else "private" for t in tracks for i in range(n)}


def test_a_better_model_ranks_first_and_every_number_has_an_interval():
    rows = rows_for("good", 0.95) + rows_for("poor", 0.55, seed=1)
    out = board.board(rows, splits_for(), CONFIG, draws=DRAWS)
    assert out["ranking"] == ["good", "poor"]
    good = out["models"]["good"]
    assert good["questions"] == 120 and good["items"] == 80 and good["unscored_rows"] == 0
    block = good["tracks"]["spam"]["all"]["raw"]["all"]
    assert block["questions"] == 80 and block["items"] == 40
    for name, value in block["metrics"].items():
        assert value["interval"] is not None, name
    assert "false_alarms" in good["tracks"]["spam"]["noul"]["raw"]["all"]["metrics"]
    assert "false_alarms" not in good["tracks"]["destek"]["choice"]["raw"]["all"]["metrics"]
    cal = good["tracks"]["spam"]["noul"]["calibrated_private"]
    assert "coverage_at_5pct_transferred" in cal["metrics"]
    assert good["temperature"]["questions"] == 60
    # Robustness and cost have no input, so they are dropped for every model.
    assert set(out["composite"]["axes_dropped"]) == {"robustness", "cost"}
    assert set(out["composite"]["weights"]) == {"intelligence", "calibration", "selective", "speed"}
    lo, hi = good["composite"]["interval"]
    assert lo <= good["composite"]["value"] <= hi


def test_axis_inputs_fill_robustness_and_cost():
    rows = rows_for("good", 0.95) + rows_for("poor", 0.55, seed=1)
    inputs = {"good": {"robustness": 0.8, "cost_usd_per_1000": 0.0},
              "poor": {"robustness": 0.5, "cost_usd_per_1000": 1.0}}  # fmt: skip
    out = board.board(rows, splits_for(), CONFIG, inputs, draws=DRAWS)
    assert out["composite"]["axes_dropped"] == {}
    assert out["models"]["poor"]["axes"]["cost"]["value"] == pytest.approx(0.5)
    assert out["models"]["good"]["axes"]["robustness"]["value"] == 0.8


def test_robustness_with_draws_carries_an_interval_into_the_composite():
    rows = rows_for("good", 0.95) + rows_for("poor", 0.55, seed=1)
    spread = np.linspace(0.6, 1.0, DRAWS)
    fixed = {"good": {"robustness": 0.8}, "poor": {"robustness": 0.5}}
    drawn = {"good": {"robustness": {"value": 0.8, "draws": spread.tolist()}},
             "poor": {"robustness": 0.5}}  # fmt: skip
    config = {**CONFIG, "axes": {"intelligence": 1.0, "robustness": 1.0}}
    plain = board.board(rows, splits_for(), config, fixed, draws=DRAWS)["models"]["good"]
    out = board.board(rows, splits_for(), config, drawn, draws=DRAWS)["models"]["good"]
    assert plain["axes"]["robustness"]["interval"] == [0.8, 0.8]
    lo, hi = out["axes"]["robustness"]["interval"]
    assert lo == pytest.approx(np.percentile(spread, 2.5))
    assert hi == pytest.approx(np.percentile(spread, 97.5))
    assert out["axes"]["robustness"]["value"] == 0.8
    # Same value, so the same composite; the robustness draws widen its interval.
    assert out["composite"]["value"] == pytest.approx(plain["composite"]["value"])
    width = out["composite"]["interval"][1] - out["composite"]["interval"][0]
    assert width > plain["composite"]["interval"][1] - plain["composite"]["interval"][0]


def test_robustness_draws_must_match_the_board():
    rows = rows_for("m", 0.9)
    short = {"m": {"robustness": {"value": 0.8, "draws": [0.8] * (DRAWS - 1)}}}
    with pytest.raises(ValueError, match="draws"):
        board.board(rows, splits_for(), CONFIG, short, draws=DRAWS)
    outside = {"m": {"robustness": {"value": 0.8, "draws": [1.5] * DRAWS}}}
    with pytest.raises(ValueError, match="0 to 1"):
        board.board(rows, splits_for(), CONFIG, outside, draws=DRAWS)


def test_main_reads_a_probes_file_as_axis_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(board, "committed_code_version", lambda root: "abc")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(r.model_dump_json() + "\n" for r in rows_for("m", 0.9)))
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits_for()))
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({**CONFIG, "axes": {"intelligence": 1.0, "robustness": 1.0}}))
    probes = tmp_path / "probes.json"
    probes.write_text(json.dumps({"script": "bench/probes.py", "models": {}, "axis_inputs": {
        "m": {"robustness": {"value": 0.7, "draws": [0.6, 0.8] * 10}}}}))  # fmt: skip
    out = tmp_path / "board.json"
    assert board.main(["--rows", str(rows_path), "--splits", str(splits_path),
                       "--weights", str(weights), "--axis-inputs", str(probes),
                       "--draws", "20", "--out", str(out)]) == 0  # fmt: skip
    written = json.loads(out.read_text(encoding="utf-8"))
    axis = written["models"]["m"]["axes"]["robustness"]
    assert axis["value"] == 0.7 and axis["interval"][0] < 0.7 < axis["interval"][1]


def test_costs_from_rows_are_per_thousand_questions():
    rows = rows_for("priced", 0.9, cost_usd=0.002)
    out = board.board(rows, splits_for(), {**CONFIG, "axes": {"cost": 1.0}}, draws=DRAWS)
    cost = out["models"]["priced"]["cost"]
    assert cost["usd_per_1000"]["value"] == pytest.approx(2.0)
    assert cost["source"] == "rows"


def test_the_gap_is_flagged_when_the_public_half_is_easier():
    rows = rows_for("leaky", 0.5, public_skill=1.0)
    gap = board.board(rows, splits_for(), CONFIG, draws=DRAWS)["models"]["leaky"]["gap"]
    assert gap["points"]["value"] > board.GAP_FLAG_POINTS and gap["flag"]
    assert gap["points"]["interval"][0] > 0
    fair = board.board(rows_for("fair", 0.8), splits_for(), CONFIG, draws=DRAWS)
    assert fair["models"]["fair"]["gap"]["flag"] is False


def test_the_gap_skips_a_track_kept_private_only():
    splits = {k: ("private" if k.startswith("spam") else v) for k, v in splits_for().items()}
    gap = board.board(rows_for("leaky", 0.5, public_skill=1.0), splits, CONFIG,
                      draws=DRAWS)["models"]["leaky"]["gap"]  # fmt: skip
    assert gap is not None and gap["tracks"] == ["destek"]


def test_a_model_missing_a_track_gets_no_composite():
    rows = rows_for("full", 0.9) + rows_for("half", 0.9, tracks=("destek",))
    out = board.board(rows, splits_for(), CONFIG, draws=DRAWS)
    assert out["models"]["half"]["composite"] is None
    assert "spam" in out["models"]["half"]["composite_note"]
    assert out["ranking"] == ["full"]


def test_the_board_does_not_depend_on_the_number_of_processes():
    rows = rows_for("m", 0.8, n=12)
    splits = splits_for(n=12)
    one = board.board(rows, splits, CONFIG, draws=10)
    two = board.board(rows, splits, CONFIG, draws=10, workers=2)
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)


def test_geometric_mean():
    axes = {"a": 0.25, "b": 1.0}
    assert board.geometric_mean(axes, {"a": 1.0, "b": 1.0}) == pytest.approx(0.5)
    assert board.geometric_mean(axes, {"a": 1.0, "b": 0.0}) == pytest.approx(0.25)
    assert board.geometric_mean({"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 1.0}) == 0.0


def test_request_latency_is_shared_by_its_questions():
    rows = [row("m", "i1", "t", q, "noul", {"type": "noul", "noul": 0.5}, True,
                latency_ms=90.0, latency_scope="request") for q in ("a", "b", "c")]  # fmt: skip
    assert set(board.per_question_latency(rows).values()) == {30.0}


def test_the_confidence_rule():
    own = row("m", "i", "t", "q", "noul", {"type": "noul", "noul": 0.9, "abstain": 0.3}, True)
    assert board.question_of(own).confidence == pytest.approx(0.7)
    assert board.own_confidence(own)
    derived = row("m", "i", "t", "q", "noul", {"type": "noul", "noul": 0.2}, False)
    assert board.question_of(derived).confidence == pytest.approx(0.8)
    assert board.question_of(derived).gold == 1
    assert not board.own_confidence(derived)
    score = row("m", "i", "t", "q", "score",
                {"type": "score", "score": 1.0, "legend": {"0": "a", "1": "b"},
                 "probabilities": {"1": 0.6, "0": 0.4}, "confidence": 0.6}, 1)  # fmt: skip
    q = board.question_of(score)
    assert q.outcomes == ("0", "1") and q.probabilities == (0.4, 0.6) and q.gold == 1
    unlabelled = row("m", "i", "t", "q", "noul", {"type": "noul", "noul": 0.2}, None)
    assert board.question_of(unlabelled) is None


def test_rows_that_cannot_be_scored_honestly_stop_the_board():
    rows = rows_for("m", 0.9)
    with pytest.raises(ValueError, match="twice"):
        board.board(rows + rows[:1], splits_for(), CONFIG, draws=DRAWS)
    splits = splits_for()
    splits.pop("destek-000")
    with pytest.raises(ValueError, match="split"):
        board.board(rows, splits, CONFIG, draws=DRAWS)
    with pytest.raises(ValueError, match="unknown axes"):
        board.board(rows, splits_for(), {**CONFIG, "axes": {"charm": 1.0}}, draws=DRAWS)


def test_models_with_one_id_and_two_revisions_are_told_apart():
    rows = rows_for("m", 0.9) + rows_for("m", 0.6, model_revision="v2")
    labels = set(board.group_models(rows))
    assert labels == {"m @ v1", "m @ v2"}


def test_main_writes_the_board_once(tmp_path, monkeypatch):
    monkeypatch.setattr(board, "committed_code_version", lambda root: "abc")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(r.model_dump_json() + "\n" for r in rows_for("m", 0.9)))
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits_for()))
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps(CONFIG))
    out = tmp_path / "step8" / "board.json"
    args = ["--rows", str(rows_path), "--splits", str(splits_path), "--weights", str(weights),
            "--draws", "20", "--out", str(out)]  # fmt: skip
    assert board.main(args) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["git_commit"] == "abc" and written["inputs"]["rows"][0]["rows"] == 120
    assert written["ranking"] == ["m"]
    with pytest.raises(SystemExit):
        board.main(args)


def items_files(tmp_path, splits):
    """The public and private items files that hold the ids of `splits`."""
    question = {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}}
    paths = []
    for half in board.SPLITS:
        path = tmp_path / f"{half}.jsonl"
        items = [{"id": i, "track": i.split("-")[0], "state": "metin", "questions": {"q": question},
                  "gold": {"q": "a"}} for i, h in splits.items() if h == half]  # fmt: skip
        path.write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")
        paths.append(path)
    return paths


def test_the_halves_can_come_from_the_two_items_files(tmp_path, monkeypatch):
    public, private = items_files(tmp_path, splits_for())
    assert board.splits_from_items(public, private) == splits_for()
    with pytest.raises(ValueError, match="both halves"):
        board.splits_from_items(public, public)

    monkeypatch.setattr(board, "committed_code_version", lambda root: "abc")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(r.model_dump_json() + "\n" for r in rows_for("m", 0.9)))
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps(CONFIG))
    out = tmp_path / "board.json"
    args = ["--rows", str(rows_path), "--items", str(public), str(private),
            "--weights", str(weights), "--draws", "20", "--out", str(out)]  # fmt: skip
    assert board.main(args) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [f["path"] for f in written["inputs"]["splits"]["from_items"]] == [
        str(public), str(private)
    ]  # fmt: skip
    assert written["ranking"] == ["m"]


def test_the_committed_weights_file_is_valid():
    from pathlib import Path

    config = json.loads((Path(__file__).resolve().parents[1] / "board_weights.json").read_text())
    assert set(config["axes"]) <= set(board.AXES)
    assert config["speed_reference_ms"] > 0 and config["cost_reference_usd_per_1000"] > 0


def test_the_owner_gold_column_scores_only_owner_decided_questions(tmp_path):
    prov = tmp_path / "provenance.jsonl"
    prov.write_text(
        json.dumps({"id": "a", "gold_source": {"q": "owner_blind", "r": "ai_rubric_agreed"}})
        + "\n"
        + json.dumps({"id": "b", "gold_source": {"q": "owner_confirmed"}})
        + "\n"
    )
    keys = board.owner_keys([prov])
    assert keys == {("a", "q"), ("b", "q")}
    rows = [
        row("m", "a", "destek", "q", "choice", choice(0.9, "x", "y"), "x"),
        row("m", "a", "destek", "r", "choice", choice(0.9, "x", "y"), "y"),
        row("m", "b", "destek", "q", "choice", choice(0.9, "x", "y"), "y"),
    ]
    col = board.owner_gold(rows, keys, 200, "m")
    assert col["questions"] == 2 and col["items"] == 2
    assert col["accuracy"]["value"] == pytest.approx(0.5)
    assert col["accuracy"]["interval"] is not None
    assert board.owner_gold(rows, set(), 200, "m") is None


def test_the_owner_check_column_scores_against_the_owners_answers(tmp_path):
    answers_file = tmp_path / "answers.json"
    answers_file.write_text(json.dumps({"answers": {"a~q": "y", "b~q": "y"}}))
    answers = board.read_owner_answers(answers_file)
    assert answers == {("a", "q"): "y", ("b", "q"): "y"}
    rows = [
        # The benchmark's gold says x on both; the owner said y.
        row("m", "a", "destek", "q", "choice", choice(0.9, "x", "y"), "x"),
        row("m", "b", "destek", "q", "choice", choice(0.9, "x", "y"), "x"),
        row("m", "c", "destek", "q", "choice", choice(0.9, "x", "y"), "x"),
    ]
    on_gold = board.owner_gold(rows, set(answers), 200, "m")
    on_owner = board.owner_gold(rows, set(answers), 200, "m", answers=answers)
    assert on_gold["accuracy"]["value"] == pytest.approx(1.0)
    assert on_owner["questions"] == 2
    assert on_owner["accuracy"]["value"] == pytest.approx(0.0)


def test_an_unranked_model_is_scored_but_not_ranked():
    rows = rows_for("good", 0.9) + rows_for("bad", 0.5, seed=1)
    out = board.board(rows, splits_for(), CONFIG, draws=DRAWS, unranked=["good"])
    assert out["ranking"] == ["bad"]
    assert out["models"]["good"]["composite"] is not None
    assert "not ranked" in out["models"]["good"]["unranked"]
    with pytest.raises(ValueError, match="not on the board"):
        board.board(rows, splits_for(), CONFIG, draws=DRAWS, unranked=["absent"])


def test_only_listed_scores_the_released_subset_and_counts_the_rest(tmp_path):
    rows = rows_for("good", 0.9)
    splits = splits_for()
    keep = dict(list(splits.items())[: len(splits) // 2])
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(keep))
    weights = tmp_path / "w.json"
    weights.write_text(json.dumps(CONFIG))
    out = tmp_path / "board.json"
    os.environ["KARAR_GIT_COMMIT"] = "test"
    try:
        board.main(["--rows", str(rows_path), "--splits", str(splits_path), "--weights",
                    str(weights), "--draws", "50", "--only-listed", "--out", str(out)])  # fmt: skip
    finally:
        del os.environ["KARAR_GIT_COMMIT"]
    result = json.loads(out.read_text())
    outside = sum(1 for r in rows if r.item_id not in keep)
    assert result["inputs"]["rows_outside_the_set"] == {"good": outside}
    assert result["models"]["good"]["items"] == len({r.item_id for r in rows if r.item_id in keep})
