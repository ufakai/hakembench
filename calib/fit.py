"""Temperature calibration fitted on the shipped file's own outputs (step 7).

Rows are the int8 file's log-probabilities per question (calib/export.py
outputs). A temperature T rescales them, softmax(logp / T), and is fitted by the
cross-entropy against the soft target. Forms, from least to most detailed:

- none;
- global: one T;
- type: one T per question type (choice, noul, score);
- bucket: one T per (type, option-count bucket), falling back to the type's T
  where a bucket has fewer than MIN_ROWS fitting rows or its T lands on a search
  bound (a temperature at the bound is the fitter's clamp, not a fit).

The form is chosen on a selection split, never on the rows it is judged on.
"""

from __future__ import annotations

import numpy as np

BOUNDS = (0.05, 20.0)
MIN_ROWS = 200
FORMS = ("none", "global", "type", "bucket")


def bucket(n_options: int) -> str:
    if n_options <= 2:
        return "2"
    if n_options <= 4:
        return "3-4"
    if n_options <= 6:
        return "5-6"
    return "7-10"


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def soft_ce(rows: list[dict], temperature: float) -> float:
    """Mean cross-entropy of the tempered distribution against the soft target."""
    total = 0.0
    for r in rows:
        p = _softmax(np.asarray(r["logprobs"]) / temperature)
        total -= float(np.dot(r["target"], np.log(np.clip(p, 1e-12, None))))
    return total / len(rows)


def fit_temperature(rows: list[dict]) -> float | None:
    """The T minimising soft cross-entropy, or None if it lands on a bound."""
    from scipy.optimize import minimize_scalar

    found = minimize_scalar(lambda log_t: soft_ce(rows, float(np.exp(log_t))),
                            bounds=tuple(np.log(BOUNDS)), method="bounded",
                            options={"xatol": 1e-4})  # fmt: skip
    t = float(np.exp(found.x))
    if t <= BOUNDS[0] * 1.01 or t >= BOUNDS[1] * 0.99:
        return None
    return t


def fit(rows: list[dict], form: str) -> dict:
    """The temperatures of one form, fitted on `rows` (each with type and n_options)."""
    if form == "none":
        return {"form": form}
    everyone = fit_temperature(rows) or 1.0
    if form == "global":
        return {"form": form, "global": everyone}
    types = {}
    for kind in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == kind]
        types[kind] = (fit_temperature(group) if len(group) >= MIN_ROWS else None) or everyone
    out = {"form": form, "global": everyone, "type": types}
    if form == "bucket":
        buckets = {}
        for kind in types:
            for b in sorted({bucket(r["n_options"]) for r in rows if r["type"] == kind}):
                group = [r for r in rows if r["type"] == kind and bucket(r["n_options"]) == b]
                t = fit_temperature(group) if len(group) >= MIN_ROWS else None
                buckets[f"{kind}:{b}"] = t if t is not None else types[kind]
        out["bucket"] = buckets
    return out


def temperature_for(calibrator: dict, kind: str, n_options: int) -> float:
    form = calibrator["form"]
    if form == "none":
        return 1.0
    if form == "global":
        return calibrator["global"]
    if form == "type":
        return calibrator["type"].get(kind, calibrator["global"])
    key = f"{kind}:{bucket(n_options)}"
    return calibrator["bucket"].get(key, calibrator["type"].get(kind, calibrator["global"]))


def apply(calibrator: dict, row: dict) -> np.ndarray:
    t = temperature_for(calibrator, row["type"], row["n_options"])
    return _softmax(np.asarray(row["logprobs"]) / t)
