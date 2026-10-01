"""HakemBench's metrics on cases whose answers are known."""

import numpy as np
import pytest

pytest.importorskip("relplot")
pytest.importorskip("scipy")

from bench import metrics  # noqa: E402
from bench.metrics import Question, Scored  # noqa: E402
from calib.selective import augrc  # noqa: E402


def q(item, probs, gold, kind="choice", outcomes=None, confidence=None):
    outcomes = outcomes or tuple("abcdefghij"[: len(probs)])
    confidence = max(probs) if confidence is None else confidence
    return Question(item, kind, tuple(outcomes), tuple(probs), gold, confidence)


def test_accuracy_brier_nll_and_padding():
    two = q("i1", (0.8, 0.2), 0)
    three = q("i2", (0.2, 0.5, 0.3), 2)
    s = Scored.build([two, three])
    assert metrics.accuracy(s) == 0.5
    # (0.2^2 + 0.2^2) and (0.2^2 + 0.5^2 + 0.7^2); padding adds nothing.
    assert metrics.brier(s) == pytest.approx((0.08 + 0.78) / 2)
    assert metrics.root_brier(s) == pytest.approx(np.sqrt(0.43 / 2))
    assert metrics.nll(s) == pytest.approx(-(np.log(0.8) + np.log(0.3)) / 2)
    assert metrics.uniform_brier(s) == pytest.approx((0.5 + 2 / 3) / 2)
    alone = Scored.build([three])
    assert metrics.brier(alone) == pytest.approx(0.78)


def test_rounded_answers_are_renormalised():
    s = Scored.build([q("i", (0.6, 0.402), 0)])
    assert s.probs[0].sum() == pytest.approx(1.0)


def test_macro_f1_counts_every_gold_or_predicted_class():
    # Gold a a b, predicted a b b: F1(a) = 2/3, F1(b) = 2/3.
    assert metrics.macro_f1(np.array([0, 1, 1]), np.array([0, 0, 1]), 3) == pytest.approx(2 / 3)
    # A class predicted but never gold still counts, with F1 0.
    assert metrics.macro_f1(np.array([2, 0]), np.array([0, 0]), 3) == pytest.approx((2 / 3) / 2)


def test_labels_of_different_types_are_different_classes():
    noul = q("i1", (0.9, 0.1), 0, kind="noul", outcomes=("true", "false"))
    choice = q("i2", (0.9, 0.1), 1, outcomes=("true", "false"))
    s = Scored.build([noul, choice])
    assert set(s.classes) == {"noul:true", "noul:false", "choice:true", "choice:false"}
    # noul:true is right (F1 1); choice:true is predicted and wrong (0); choice:false missed (0).
    assert metrics.block_macro_f1(s) == pytest.approx(1 / 3)


def test_smooth_ece_is_small_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 4000)
    hit = (rng.uniform(size=4000) < p).astype(float)
    assert metrics.smooth_ece(p, hit) < 0.03
    assert metrics.smooth_ece(np.full(4000, 0.99), hit) > 0.15


def test_classwise_ece_reads_each_level():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(600):
        p = rng.dirichlet(np.ones(3))
        rows.append(q(f"i{i}", tuple(p), int(rng.choice(3, p=p)), kind="score",
                      outcomes=("0", "1", "2")))  # fmt: skip
    calibrated = Scored.build(rows)
    sharp = Scored.build([q(r.item_id, tuple(np.eye(3)[int(np.argmax(r.probabilities))]), r.gold,
                            kind="score", outcomes=("0", "1", "2")) for r in rows])  # fmt: skip
    assert metrics.classwise_smooth_ece(calibrated) < metrics.classwise_smooth_ece(sharp)


def test_ties_are_broken_in_expectation():
    confidence = np.array([0.9, 0.5, 0.5, 0.1])
    first = metrics.selective_augrc(confidence, np.array([0.0, 1.0, 0.0, 1.0]))
    second = metrics.selective_augrc(confidence, np.array([0.0, 0.0, 1.0, 1.0]))
    assert first == pytest.approx(second)
    # Without ties it is calib/selective.py's AUGRC.
    distinct = np.array([0.9, 0.6, 0.5, 0.1])
    loss = np.array([0.0, 1.0, 0.0, 1.0])
    assert metrics.selective_augrc(distinct, loss) == pytest.approx(augrc(distinct, loss))
    assert metrics.worst_augrc(4) == pytest.approx(augrc(distinct, np.ones(4)))


def test_a_threshold_takes_a_whole_tie_group():
    # Answering the first two rows alone would have risk 0, but they tie with a
    # wrong one, so the only honest thresholds are 0.9 (risk 0) and 0.5 (risk 1/3).
    confidence = np.array([0.9, 0.5, 0.5])
    loss = np.array([0.0, 0.0, 1.0])
    assert metrics.threshold_at_risk(confidence, loss, 0.05) == 0.9
    out = metrics.coverage_at_risk(confidence, loss, 0.05)
    assert out["coverage"] == pytest.approx(1 / 3) and out["risk"] == 0.0
    assert metrics.threshold_at_risk(confidence, np.ones(3), 0.05) is None


def test_false_alarms_and_misses():
    out = metrics.error_direction(["spam", "spam", "ham", "ham"], ["spam", "ham", "spam", "ham"],
                                  positive="spam")  # fmt: skip
    assert out == {"false_alarms": 1, "misses": 1, "false_alarm_rate": 0.5, "miss_rate": 0.5,
                   "negatives": 2, "positives": 2}  # fmt: skip


def test_error_direction_in_a_block_reads_only_binary_questions_with_the_class():
    rows = [q("i1", (0.9, 0.1), 0, outcomes=("spam", "ham")),
            q("i2", (0.9, 0.1), 1, outcomes=("spam", "ham")),
            q("i3", (0.2, 0.8), 0, outcomes=("spam", "ham")),
            q("i4", (0.5, 0.3, 0.2), 1, outcomes=("spam", "ham", "other"))]  # fmt: skip
    out = metrics.point(Scored.build(rows), positive="spam")
    assert out["false_alarms"] == 1 and out["misses"] == 1
    assert out["false_alarm_rate"] == 1.0 and out["miss_rate"] == 0.5


def test_temperature_one_changes_nothing_and_a_fit_recovers_a_known_one():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(1500):
        logits = rng.normal(size=3) * 1.5
        truth = np.exp(logits) / np.exp(logits).sum()
        shown = np.exp(2 * logits) / np.exp(2 * logits).sum()
        rows.append(q(f"i{i}", tuple(shown), int(rng.choice(3, p=truth))))
    s = Scored.build(rows)
    assert metrics.apply_temperature(s, 1.0) == pytest.approx(s.probs)
    t = metrics.fit_temperature(s)
    assert t == pytest.approx(2.0, rel=0.15)
    flat = metrics.apply_temperature(s, t)
    assert metrics.nll(s.with_probabilities(flat, flat.max(1))) < metrics.nll(s)


def test_temperature_keeps_padding_and_zeros_at_zero():
    s = Scored.build([q("i1", (1.0, 0.0), 0), q("i2", (0.5, 0.3, 0.2), 1)])
    out = metrics.apply_temperature(s, 2.0)
    assert out[0].tolist() == [1.0, 0.0, 0.0]
    assert out[1, :3].sum() == pytest.approx(1.0)


def test_point_has_every_metric():
    rows = [q(f"i{i}", (0.7, 0.3) if i % 3 else (0.4, 0.6), 0, kind="score",
              outcomes=("0", "1")) for i in range(30)]  # fmt: skip
    out = metrics.point(Scored.build(rows), classwise=True, thresholds={0.05: 0.65, 0.01: None})
    for name in ("accuracy", "macro_f1", "brier", "root_brier", "brier_normalised", "nll",
                 "smooth_ece", "classwise_ece", "augrc", "augrc_normalised", "coverage_at_5pct",
                 "risk_at_5pct", "coverage_at_1pct", "coverage_at_5pct_transferred",
                 "risk_at_1pct_transferred"):  # fmt: skip
        assert name in out, name
    assert out["coverage_at_5pct_transferred"] == pytest.approx(20 / 30)
    assert out["coverage_at_1pct_transferred"] == 0.0


def test_bootstrap_draws_whole_items_and_is_reproducible():
    rows = []
    for i in range(12):
        rows += [q(f"i{i:02d}", (0.8, 0.2), 0), q(f"i{i:02d}", (0.3, 0.7), 0)]
    s = Scored.build(rows)
    for picked in metrics.resamples(s, draws=20, seed=3):
        counts = np.bincount(s.item[picked], minlength=12)
        assert (counts % 2 == 0).all()
    report, draws = metrics.evaluate(s, draws=100, seed=3)
    again, _ = metrics.evaluate(s, draws=100, seed=3)
    assert report == again
    assert report["questions"] == 24 and report["items"] == 12
    # Every item is half right, so every draw has accuracy one half.
    assert report["metrics"]["accuracy"] == {"value": 0.5, "interval": [0.5, 0.5]}
    assert len(draws["brier"]) == 100
    lo, hi = report["metrics"]["nll"]["interval"]
    assert lo <= report["metrics"]["nll"]["value"] <= hi


def test_a_block_needs_distributions():
    with pytest.raises(ValueError):
        Scored.build([])
    with pytest.raises(ValueError):
        Scored.build([q("i", (0.5, 0.5), 2)])
    with pytest.raises(ValueError):
        Scored.build([q("i", (-0.1, 1.1), 0)])
