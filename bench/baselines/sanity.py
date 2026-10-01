"""Sanity checks on baseline rows, no gold used: sums, keys, spread, and how close to uniform.

    python -m bench.baselines.sanity --rows results/step9/smoke/*.jsonl --items <items file> ...

Per rows file: the rows' count; whether each distribution sums to one and
covers exactly the question's options or levels; the mean maximum probability
against the uniform 1/k; the mean normalised entropy (1 is uniform); how many
distributions sit within 0.02 of uniform in every outcome; how often each option
position wins (a model that always picks the first option shows here); and the
noul mean and spread. Prints JSON; --out writes it. Gold labels are never read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from bench.harness.items import load_items
from schema.questions import PROBABILITY_SUM_TOLERANCE


def outcomes(question: dict) -> list[str]:
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(k) for k in range(len(question["criteria"]))]


def check(rows: list[dict], questions: dict[tuple[str, str], dict]) -> dict:
    report: dict = {"rows": len(rows), "bad_sum": 0, "bad_keys": 0}
    by_type: dict[str, dict] = {}
    for row in rows:
        question = questions[(row["item_id"], row["question_id"])]
        answer = row["answer"]
        kind = row["question_type"]
        stats = by_type.setdefault(kind, {"n": 0, "max_p": 0.0, "uniform_max_p": 0.0,
                                          "entropy": 0.0, "near_uniform": 0,
                                          "winning_position": Counter(), "noul": []})  # fmt: skip
        stats["n"] += 1
        if kind == "noul":
            stats["noul"].append(answer["noul"])
            continue
        probs = answer["probabilities"]
        keys = outcomes(question)
        if set(probs) != set(keys):
            report["bad_keys"] += 1
            continue
        if abs(math.fsum(probs.values()) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            report["bad_sum"] += 1
        p = [probs[k] for k in keys]
        k = len(p)
        stats["max_p"] += max(p)
        stats["uniform_max_p"] += 1.0 / k
        stats["entropy"] += -sum(x * math.log(x) for x in p if x > 0) / math.log(k)
        stats["near_uniform"] += all(abs(x - 1.0 / k) < 0.02 for x in p)
        stats["winning_position"][p.index(max(p))] += 1
    for kind, stats in by_type.items():
        n = stats.pop("n")
        noul = stats.pop("noul")
        out = {"n": n}
        if kind == "noul":
            mean = sum(noul) / n
            out["noul_mean"] = round(mean, 4)
            out["noul_sd"] = round(math.sqrt(sum((x - mean) ** 2 for x in noul) / n), 4)
            out["noul_above_half"] = sum(x > 0.5 for x in noul)
        else:
            out["mean_max_p"] = round(stats["max_p"] / n, 4)
            out["mean_uniform_max_p"] = round(stats["uniform_max_p"] / n, 4)
            out["mean_normalised_entropy"] = round(stats["entropy"] / n, 4)
            out["near_uniform"] = stats["near_uniform"]
            out["winning_position"] = dict(sorted(stats["winning_position"].items()))
        report[kind] = out
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.baselines.sanity")
    parser.add_argument("--rows", type=Path, nargs="+", required=True)
    parser.add_argument("--items", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    questions = {}
    for path in args.items:
        for item in load_items(path):
            for qid, q in item.questions.items():
                questions[(item.id, qid)] = q.model_dump(mode="json")
    result = {}
    for path in args.rows:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        result[str(path)] = check(rows, questions)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        if args.out.exists():
            print(f"{args.out} exists; write to a new file", file=sys.stderr)
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
