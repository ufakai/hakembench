"""The surface baseline: one cue per question, fitted on the public half only."""

from bench import surface_baseline as S
from bench.harness.items import Item


def item(uid, text, gold, track="spam"):
    return Item(id=uid, track=track, state=text,
                questions={"q": {"type": "noul", "instructions": "?", "criteria": {}}},
                gold={"q": gold})  # fmt: skip


def test_the_cue_that_separates_the_public_gold_is_found_and_applied():
    public = [item(f"p{n}", f"mesaj {n}?" if n % 2 else f"mesaj {n}", bool(n % 2))
              for n in range(40)]  # fmt: skip
    private = [item("x1", "yeni mesaj?", True), item("x2", "yeni mesaj", False)]
    rows, cues = S.predictions(public, private)
    assert cues["spam/q"] == "question_mark"
    by_id = {i.id: a for i, _, a in rows}
    assert by_id["x1"]["noul"] > 0.9 and by_id["x2"]["noul"] < 0.1
    assert len(rows) == 42


def test_no_cue_falls_back_to_the_gold_mix():
    public = [item(f"p{n}", "aynı metin", n < 30) for n in range(40)]
    rows, cues = S.predictions(public, [])
    assert cues["spam/q"] == "none"
    assert rows[0][2]["noul"] == (30 + 1) / (40 + 2)


def test_the_open_layout_gives_the_two_parts_in_file_order(tmp_path):
    import json

    rows = [item("a", "bir?", True), item("b", "iki", False), item("c", "üç?", True)]
    items = tmp_path / "items.jsonl"
    items.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"a": "public", "b": "private", "c": "public"}))
    public, private = S.open_parts(items, splits)
    assert [i.id for i in public] == ["a", "c"] and [i.id for i in private] == ["b"]
