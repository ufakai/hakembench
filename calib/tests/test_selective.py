"""Selective prediction metrics on cases whose answers are known."""

import numpy as np
import pytest

from calib.selective import at_threshold, augrc, oracle_augrc, scores, threshold_at_risk


def test_a_confidence_that_ranks_errors_last_scores_better():
    loss = np.array([0.0, 0.0, 1.0, 1.0])
    good = np.array([0.9, 0.8, 0.2, 0.1])
    bad = good[::-1].copy()
    assert augrc(good, loss) < augrc(bad, loss)
    assert augrc(good, loss) == pytest.approx(oracle_augrc(loss))
    # Generalised risk at coverage 1..4: 0, 0, 1/4, 2/4; their mean is 3/16.
    assert augrc(good, loss) == pytest.approx(3 / 16)


def test_the_threshold_read_on_one_split_is_applied_to_another():
    fit_conf, fit_loss = np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.0, 0.0, 0.0, 1.0])
    t = threshold_at_risk(fit_conf, fit_loss, 0.05)
    assert t == 0.7
    out = at_threshold(np.array([0.95, 0.75, 0.5]), np.array([0.0, 1.0, 0.0]), t)
    assert out == {"coverage": pytest.approx(2 / 3), "risk": 0.5}
    assert threshold_at_risk(fit_conf, np.ones(4), 0.05) is None


def test_baseline_scores_agree_on_a_confident_and_an_unsure_question():
    sure, unsure = scores([5.0, 0.0, 0.0]), scores([0.1, 0.0, -0.1])
    for name in sure:
        assert sure[name] > unsure[name], name
    assert scores([1.0, float("-inf"), 0.0])["max_probability"] == pytest.approx(
        np.exp(1) / (np.exp(1) + 1)
    )
