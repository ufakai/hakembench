"""The baseline sanity report: sums, keys, spread. No gold is read."""

from bench.baselines.sanity import check

QUESTIONS = {
    ("a", "tür"): {"type": "choice", "criteria": {"x": None, "y": None}},
    ("a", "puan"): {"type": "score", "criteria": ["az", "orta", "çok"]},
    ("a", "evet"): {"type": "noul"},
}


def row(qid, kind, answer):
    return {"item_id": "a", "question_id": qid, "question_type": kind, "answer": answer}


def test_flags_bad_sums_and_keys_and_measures_spread():
    rows = [
        row("tür", "choice", {"probabilities": {"x": 0.5, "y": 0.5}}),
        row("tür", "choice", {"probabilities": {"x": 0.9, "y": 0.3}}),
        row("tür", "choice", {"probabilities": {"x": 0.5, "z": 0.5}}),
        row("puan", "score", {"probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}}),
        row("evet", "noul", {"noul": 0.2}),
        row("evet", "noul", {"noul": 0.8}),
    ]
    report = check(rows, QUESTIONS)
    assert (report["rows"], report["bad_sum"], report["bad_keys"]) == (6, 1, 1)
    assert report["choice"]["near_uniform"] == 1
    assert report["choice"]["winning_position"] == {0: 2}
    assert report["score"]["winning_position"] == {2: 1}
    assert report["noul"] == {"n": 2, "noul_mean": 0.5, "noul_sd": 0.3, "noul_above_half": 1}
