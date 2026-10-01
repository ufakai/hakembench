"""HakemBench's metrics: pure functions over stored probabilities and gold labels.

One scored question is a probability distribution over its outcomes (the
answer as a results row stores it), the index of the gold outcome, the
confidence a selective policy ranks it by, and the item it belongs to.
Conventions, stated once:

  prediction     the most probable outcome; ties go to the first in the
                 answer's order.
  macro F1       unweighted mean of per-class F1 over every class that is a
                 gold or a predicted label in the block. Labels are named
                 "type:outcome", so a block that mixes question types never
                 merges one type's "true" with another's.
  Brier          sum over outcomes of the squared error, 0 to 2 (calib/metrics.py).
  root Brier     sqrt(Brier / 2), 0 to 1.
  Brier, normalised
                 Brier over the Brier of the uniform guess, 1 - 1/K per
                 question: 1.0 is no skill. The uniform guess needs no
                 fitting, so it is the same reference for every model.
  NLL            mean -log p(gold), with p floored at 1e-12.
  smooth ECE     top-label smooth ECE (arXiv 2309.12236), through relplot in
                 one function, smooth_ece, so it can be swapped.
  classwise ECE  for score questions: the mean over levels k of the smooth
                 ECE of p_k against gold == k, over the questions that have
                 level k.
  AUGRC          calib/selective.py (arXiv 2407.01032) on the 0/1 error.
                 Ties in confidence are broken in expectation: tied
                 questions share their mean error, which is the average over
                 every order of the tie. AUGRC, normalised, divides by
                 (N + 1) / 2N, the AUGRC of a model wrong on every question.
  coverage at r  the largest share of questions answered, at a confidence
                 threshold, with selective risk at or under r. A threshold
                 always takes a whole tie group. In-sample by default; with a
                 threshold read on another split (the public half), the
                 coverage and realised risk of that threshold here.
  false alarms and misses
                 for binary safety questions, with the positive class named
                 by the caller: a false alarm is a negative predicted
                 positive, a miss a positive predicted negative. Rates are
                 over the gold negatives and the gold positives.
  temperature    softmax(log p / T), which equals softmax(logits / T); one T
                 per model, fitted by cross-entropy on the public half
                 (calib/fit.py) and applied to the private half.
  intervals      2.5 and 97.5 percentiles of 2,000 bootstrap resamples over
                 items: an item's questions are drawn together. Items are
                 sorted by id before drawing, so two models scored on the same
                 items with the same seed see the same resamples.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from calib.selective import at_threshold, augrc

DRAWS = 2000
RISKS = (0.05, 0.01)
NLL_FLOOR = 1e-12


@dataclass(frozen=True)
class Question:
    """One scored question, as the board reads it from a results row."""

    item_id: str
    type: str  # "choice", "score" or "noul"
    outcomes: tuple[str, ...]
    probabilities: tuple[float, ...]
    gold: int  # index into outcomes
    confidence: float


@dataclass(frozen=True)
class Scored:
    """A block of questions as arrays, padded to the widest question.

    Padding holds probability 0 and is never gold, so it adds nothing to any
    sum and is never predicted.
    """

    probs: np.ndarray  # (n, K) float64
    valid: np.ndarray  # (n, K) bool
    gold: np.ndarray  # (n,) int
    confidence: np.ndarray  # (n,) float64
    item: np.ndarray  # (n,) int, the item's index in `items`
    items: tuple[str, ...]  # item ids, sorted
    labels: np.ndarray  # (n, K) int, the class id of each outcome, -1 in padding
    classes: tuple[str, ...]  # class names, "type:outcome"
    types: tuple[str, ...]  # (n,) question types

    @classmethod
    def build(cls, questions: Sequence[Question]) -> Scored:
        if not questions:
            raise ValueError("a block needs at least one question")
        width = max(len(q.outcomes) for q in questions)
        n = len(questions)
        probs = np.zeros((n, width))
        valid = np.zeros((n, width), dtype=bool)
        labels = np.full((n, width), -1, dtype=np.int64)
        items = tuple(sorted({q.item_id for q in questions}))
        item_index = {item: i for i, item in enumerate(items)}
        class_index: dict[str, int] = {}
        for row, q in enumerate(questions):
            if len(q.outcomes) != len(q.probabilities) or len(q.outcomes) < 2:
                raise ValueError(f"{q.item_id}: outcomes and probabilities do not match")
            if not 0 <= q.gold < len(q.outcomes):
                raise ValueError(f"{q.item_id}: gold index {q.gold} is out of range")
            p = np.asarray(q.probabilities, dtype=np.float64)
            if (p < 0).any() or not np.isfinite(p).all() or p.sum() <= 0:
                raise ValueError(f"{q.item_id}: probabilities are not a distribution")
            # Stored answers may be rounded; every metric reads a distribution
            # that sums to one exactly.
            probs[row, : len(p)] = p / p.sum()
            valid[row, : len(p)] = True
            for k, outcome in enumerate(q.outcomes):
                name = f"{q.type}:{outcome}"
                labels[row, k] = class_index.setdefault(name, len(class_index))
        return cls(
            probs=probs,
            valid=valid,
            gold=np.array([q.gold for q in questions], dtype=np.int64),
            confidence=np.array([q.confidence for q in questions], dtype=np.float64),
            item=np.array([item_index[q.item_id] for q in questions], dtype=np.int64),
            items=items,
            labels=labels,
            classes=tuple(class_index),
            types=tuple(q.type for q in questions),
        )

    def __len__(self) -> int:
        return len(self.gold)

    def take(self, rows: np.ndarray) -> Scored:
        """The questions at `rows`, repeats allowed (a bootstrap draw)."""
        return Scored(self.probs[rows], self.valid[rows], self.gold[rows],
                      self.confidence[rows], self.item[rows], self.items, self.labels[rows],
                      self.classes, tuple(self.types[i] for i in rows))  # fmt: skip

    def where(self, mask: np.ndarray) -> Scored:
        return self.take(np.nonzero(mask)[0])

    def with_probabilities(self, probs: np.ndarray, confidence: np.ndarray) -> Scored:
        return Scored(probs, self.valid, self.gold, confidence, self.item, self.items,
                      self.labels, self.classes, self.types)  # fmt: skip


# Pieces ---------------------------------------------------------------------


def predicted(s: Scored) -> np.ndarray:
    return np.where(s.valid, s.probs, -1.0).argmax(axis=1)


def correct(s: Scored) -> np.ndarray:
    return (predicted(s) == s.gold).astype(np.float64)


def max_probability(s: Scored) -> np.ndarray:
    return np.where(s.valid, s.probs, 0.0).max(axis=1)


def accuracy(s: Scored) -> float:
    return float(correct(s).mean())


def macro_f1(predicted_ids: np.ndarray, gold_ids: np.ndarray, n_classes: int) -> float:
    """Macro F1 over the classes that appear as a gold or a predicted label."""
    true_pos = np.bincount(gold_ids[predicted_ids == gold_ids], minlength=n_classes)
    gold_count = np.bincount(gold_ids, minlength=n_classes)
    pred_count = np.bincount(predicted_ids, minlength=n_classes)
    present = (gold_count + pred_count) > 0
    f1 = 2 * true_pos[present] / (gold_count[present] + pred_count[present])
    return float(f1.mean())


def block_macro_f1(s: Scored) -> float:
    rows = np.arange(len(s))
    return macro_f1(s.labels[rows, predicted(s)], s.labels[rows, s.gold], len(s.classes))


def _one_hot(s: Scored) -> np.ndarray:
    target = np.zeros_like(s.probs)
    target[np.arange(len(s)), s.gold] = 1.0
    return target


def brier(s: Scored) -> float:
    return float(((s.probs - _one_hot(s)) ** 2).sum(axis=1).mean())


def root_brier(s: Scored) -> float:
    return float(np.sqrt(brier(s) / 2.0))


def uniform_brier(s: Scored) -> float:
    """The Brier of the uniform guess, 1 - 1/K per question, averaged."""
    return float((1.0 - 1.0 / s.valid.sum(axis=1)).mean())


def nll(s: Scored) -> float:
    gold_p = s.probs[np.arange(len(s)), s.gold]
    return float(-np.log(np.clip(gold_p, NLL_FLOOR, None)).mean())


def smooth_ece(confidence: np.ndarray, outcome: np.ndarray) -> float:
    """Smooth ECE of a predicted probability against a 0/1 outcome (arXiv 2309.12236).

    The only call into relplot, so the implementation can be replaced here alone.
    """
    import relplot

    return float(relplot.smECE(np.asarray(confidence, dtype=np.float64),
                               np.asarray(outcome, dtype=np.float64)))  # fmt: skip


def top_label_smooth_ece(s: Scored) -> float:
    return smooth_ece(max_probability(s), correct(s))


def classwise_smooth_ece(s: Scored) -> float:
    """Mean over outcome positions (score levels) of each level's smooth ECE."""
    values = []
    for k in range(s.probs.shape[1]):
        has = s.valid[:, k]
        if has.any():
            values.append(smooth_ece(s.probs[has, k], (s.gold[has] == k).astype(np.float64)))
    return float(np.mean(values))


def tie_averaged(confidence: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """Each question's loss replaced by the mean loss of its confidence tie group."""
    _, group = np.unique(confidence, return_inverse=True)
    sums = np.bincount(group, weights=loss)
    counts = np.bincount(group)
    return (sums / counts)[group]


def selective_augrc(confidence: np.ndarray, loss: np.ndarray) -> float:
    return augrc(confidence, tie_averaged(confidence, loss))


def worst_augrc(n: int) -> float:
    """The AUGRC of a model wrong on every one of n questions."""
    return (n + 1) / (2 * n)


def threshold_at_risk(confidence: np.ndarray, loss: np.ndarray, risk: float) -> float | None:
    """The lowest threshold whose selective risk is at or under `risk`, whole ties only.

    calib/selective.py's version may stop inside a tie group, which a
    threshold cannot do; here only the last question of each group counts.
    """
    order = np.argsort(-confidence, kind="stable")
    ranked = confidence[order]
    selective = np.cumsum(loss[order]) / np.arange(1, len(loss) + 1)
    group_end = np.r_[ranked[1:] != ranked[:-1], True]
    ok = np.nonzero(group_end & (selective <= risk))[0]
    if not len(ok):
        return None
    return float(ranked[ok.max()])


def coverage_at_risk(confidence: np.ndarray, loss: np.ndarray, risk: float) -> dict:
    """Coverage and realised risk at the threshold read on these questions."""
    threshold = threshold_at_risk(confidence, loss, risk)
    return {"threshold": threshold, **at_threshold(confidence, loss, threshold)}


def error_direction(predicted_labels: Sequence[str], gold_labels: Sequence[str],
                    positive: str) -> dict:  # fmt: skip
    """False alarms and misses of a binary decision whose positive class is `positive`."""
    pred = np.asarray(predicted_labels) == positive
    gold = np.asarray(gold_labels) == positive
    false_alarms, misses = int((pred & ~gold).sum()), int((~pred & gold).sum())
    negatives, positives = int((~gold).sum()), int(gold.sum())
    return {"false_alarms": false_alarms, "misses": misses,
            "false_alarm_rate": false_alarms / negatives if negatives else None,
            "miss_rate": misses / positives if positives else None,
            "negatives": negatives, "positives": positives}  # fmt: skip


def apply_temperature(s: Scored, temperature: float) -> np.ndarray:
    """softmax(log p / T) per question, padding kept at zero."""
    with np.errstate(divide="ignore"):
        z = np.where(s.valid & (s.probs > 0), np.log(s.probs), -np.inf) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(s: Scored) -> float | None:
    """One temperature by cross-entropy on these questions (calib/fit.py); None at a bound."""
    from calib.fit import fit_temperature as fit

    rows = []
    with np.errstate(divide="ignore"):
        for i in range(len(s)):
            keep = s.valid[i]
            target = np.zeros(int(keep.sum()))
            target[s.gold[i]] = 1.0
            rows.append({"logprobs": np.log(s.probs[i, keep]), "target": target})
    return fit(rows)


def _direction(s: Scored, positive: str) -> dict:
    """Error direction over the block's two-outcome questions that have `positive`."""
    rows = np.arange(len(s))
    outcome = [name.split(":", 1)[1] for name in s.classes]
    pred = [outcome[c] for c in s.labels[rows, predicted(s)]]
    gold = [outcome[c] for c in s.labels[rows, s.gold]]
    has = [
        i
        for i in rows
        if s.valid[i].sum() == 2 and positive in {outcome[c] for c in s.labels[i] if c >= 0}
    ]
    if not has:
        return {}
    direction = error_direction([pred[i] for i in has], [gold[i] for i in has], positive)
    return {key: direction[key]
            for key in ("false_alarms", "misses", "false_alarm_rate", "miss_rate")}  # fmt: skip


# One block's numbers ------------------------------------------------------------


def point(s: Scored, *, positive: str | None = None, classwise: bool = False,
          thresholds: dict[float, float | None] | None = None) -> dict:  # fmt: skip
    """Every metric of one block.

    positive: the positive outcome of a binary safety question; error
    direction is reported over the block's two-outcome questions that have it.
    classwise: add the classwise ECE (score blocks).
    thresholds: per risk, a threshold read elsewhere; adds the coverage and
    realised risk it gives here.
    """
    loss = 1.0 - correct(s)
    b = brier(s)
    a = selective_augrc(s.confidence, loss)
    out: dict[str, float | None] = {
        "accuracy": accuracy(s),
        "macro_f1": block_macro_f1(s),
        "brier": b,
        "root_brier": float(np.sqrt(b / 2.0)),
        "brier_normalised": b / uniform_brier(s),
        "nll": nll(s),
        "smooth_ece": top_label_smooth_ece(s),
        "augrc": a,
        "augrc_normalised": a / worst_augrc(len(s)),
    }
    if classwise:
        out["classwise_ece"] = classwise_smooth_ece(s)
    for r in RISKS:
        name = f"{round(r * 100)}pct"
        at = coverage_at_risk(s.confidence, loss, r)
        out[f"coverage_at_{name}"] = at["coverage"]
        out[f"risk_at_{name}"] = at["risk"]
        if thresholds is not None:
            # A threshold read on another split; None there means nothing is answered.
            moved = at_threshold(s.confidence, loss, thresholds.get(r))
            out[f"coverage_at_{name}_transferred"] = moved["coverage"]
            out[f"risk_at_{name}_transferred"] = moved["risk"]
    if positive is not None:
        out.update(_direction(s, positive))
    return out


def resamples(s: Scored, draws: int = DRAWS, seed: int = 0) -> Iterator[np.ndarray]:
    """Row indices of each bootstrap draw, drawing whole items with replacement."""
    rng = np.random.default_rng(seed)
    order = np.argsort(s.item, kind="stable")
    starts = np.searchsorted(s.item[order], np.arange(len(s.items) + 1))
    by_item = [order[starts[i] : starts[i + 1]] for i in range(len(s.items))]
    present = [i for i, rows in enumerate(by_item) if len(rows)]
    for _ in range(draws):
        picks = rng.choice(present, size=len(present), replace=True)
        yield np.concatenate([by_item[i] for i in picks])


def interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return np.percentile(finite, [2.5, 97.5]).tolist()


def evaluate(s: Scored, statistic: Callable[[Scored], dict] = point, draws: int = DRAWS,
             seed: int = 0) -> tuple[dict, dict[str, np.ndarray]]:  # fmt: skip
    """Every number of `statistic` on the block, with its bootstrap interval.

    Returns the report and the draws per metric (NaN where a draw had no
    value, e.g. no gold negatives), which the board combines across tracks.
    """
    values = statistic(s)
    collected: dict[str, list[float]] = {name: [] for name in values}
    for rows in resamples(s, draws, seed):
        drawn = statistic(s.take(rows))
        for name in collected:
            v = drawn.get(name)
            collected[name].append(np.nan if v is None else float(v))
    arrays = {name: np.asarray(v, dtype=np.float64) for name, v in collected.items()}
    report = {"questions": len(s), "items": len(set(s.item.tolist())),
              "metrics": {name: {"value": values[name], "interval": interval(arrays[name])}
                          for name in values}}  # fmt: skip
    return report, arrays
