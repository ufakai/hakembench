"""Temperature calibration: recovers a known temperature, falls back where data is thin."""

import numpy as np
import pytest

from calib.fit import MIN_ROWS, apply, fit, fit_temperature, soft_ce


def rows_with(temperature: float, n: int, kind: str = "choice", k: int = 3, seed: int = 0):
    """Rows whose true distribution is the logits at temperature 1, shown sharpened."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        logits = rng.normal(size=k) * 2
        true = np.exp(logits) / np.exp(logits).sum()
        shown = logits / temperature  # an overconfident model when temperature < 1
        out.append({"logprobs": shown - np.log(np.exp(shown).sum()), "target": true,
                    "type": kind, "n_options": k})  # fmt: skip
    return out


def test_a_known_temperature_is_recovered():
    # Logits sharpened by 1 / 0.5 are undone by a calibration temperature of 2.
    rows = rows_with(0.5, 400)
    assert fit_temperature(rows) == pytest.approx(2.0, rel=0.02)
    assert soft_ce(rows, 2.0) < soft_ce(rows, 1.0)


def test_thin_buckets_fall_back_to_their_type():
    rows = rows_with(0.5, MIN_ROWS + 50, k=3) + rows_with(2.0, 20, k=8, seed=1)
    calibrator = fit(rows, "bucket")
    assert calibrator["bucket"]["choice:3-4"] == pytest.approx(2.0, rel=0.05)
    assert calibrator["bucket"]["choice:7-10"] == calibrator["type"]["choice"]


def test_none_leaves_the_distribution_alone():
    row = rows_with(0.5, 1)[0]
    assert np.allclose(apply(fit([row], "none"), row), np.exp(row["logprobs"]))
