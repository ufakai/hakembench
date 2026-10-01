"""The open-jev adapter on a fake OpenJev: the mapping, the answers, the refusals. No network."""

from types import SimpleNamespace

import pytest

from bench.adapters.base import AdapterError
from bench.adapters.openjev import REPO, REVISION, OpenJevAdapter, _only, model_question
from schema.api import Request, check_response

STATE = "Kargo on gündür gelmedi, param iade edilsin."
QUESTIONS = {
    "konu": {"type": "choice", "instructions": "Mesaj hangi konuda?",
             "criteria": {"kargo": "teslimat gecikmesi", "iade": None, "diğer": "başka her şey"}},
    "öfke": {"type": "score", "instructions": "Ne kadar öfkeli?",
             "criteria": ["sakin", "rahatsız", "öfkeli"]},
    "iade": {"type": "noul", "instructions": "İade istiyor mu?",
             "criteria": {"true": "evet", "false": "hayır"}},
}  # fmt: skip


class Collator:
    def __init__(self, limit):
        self.limit = limit
        self.max_state = 5

    def encode_one(self, state, questions):
        n = len(state.split()) + sum(2 + len(q.options) for q in questions)
        if n > self.limit:
            raise ValueError(f"sequence {n} > max_len {self.limit}")
        return list(range(n)), [], [], []


class FakeOpenJev:
    """The parts of typed_decisions.open_jev.OpenJev the adapter touches."""

    def __init__(self, limit=512, score_probs=(0.1, 0.3, 0.6)):
        self.collator = Collator(limit)
        self.config = {"max_state_tokens": 256, "max_len": 512, "temperature": 1.05}
        self.tok = lambda text, add_special_tokens=False: {"input_ids": text.split()}
        self.score_probs = score_probs
        self.calls = []

    @staticmethod
    def _question(i, q):
        options = ["no", "yes"] if q["type"] == "noul" else list(q["options"])
        return SimpleNamespace(qid=f"q{i}", kind=q["type"], options=options)

    def decide(self, state, questions):
        self.calls.append((state, questions))
        out = []
        for q in questions:
            if q["type"] == "noul":
                out.append({"noul": 0.8})
            elif q["type"] == "choice":
                p = [0.2, 0.5, 0.3]
                out.append({"choice": q["options"][1], "confidence": 0.5,
                            "probabilities": dict(zip(q["options"], p, strict=True))})  # fmt: skip
            else:
                p = list(self.score_probs)
                out.append({"score": sum(i * v for i, v in enumerate(p)), "confidence": max(p),
                            "probabilities": dict(zip(q["options"], p, strict=True))})  # fmt: skip
        return out


def request(questions=QUESTIONS) -> Request:
    return Request.model_validate({"state": STATE, "model": "x", "questions": questions})


def test_questions_go_in_the_models_format_keys_levels_and_instructions():
    model = FakeOpenJev()
    OpenJevAdapter(model=model).answer(request())
    (state, sent), = model.calls  # fmt: skip
    assert state == STATE
    assert sent == [
        {"type": "choice", "instructions": "Mesaj hangi konuda?",
         "options": ["kargo", "iade", "diğer"]},
        {"type": "score", "instructions": "Ne kadar öfkeli?",
         "options": ["sakin", "rahatsız", "öfkeli"]},
        {"type": "noul", "instructions": "İade istiyor mu?"},
    ]  # fmt: skip


def test_answers_follow_the_contract_with_the_models_numbers():
    req = request()
    result = OpenJevAdapter(model=FakeOpenJev()).answer(req)
    check_response(req, result.response)
    answers = result.response.answers
    assert answers["konu"].choice == "iade"
    assert answers["konu"].probabilities == {"kargo": 0.2, "iade": 0.5, "diğer": 0.3}
    assert answers["konu"].confidence == 0.5
    assert answers["öfke"].probabilities == {"0": 0.1, "1": 0.3, "2": 0.6}
    assert answers["öfke"].score == pytest.approx(1.5)
    assert answers["iade"].noul == pytest.approx(0.8)
    # decide() reports max(p) as its confidence, so the rows say so.
    assert result.confidence_source == {"konu": "max_probability", "öfke": "max_probability"}
    assert result.diagnostics["konu"]["state_tokens"] == len(STATE.split())
    assert result.diagnostics["konu"]["state_tokens_read"] == 5


def test_a_float32_rounding_past_the_last_level_is_held_on_the_scale():
    req = request({"öfke": QUESTIONS["öfke"]})
    result = OpenJevAdapter(model=FakeOpenJev(score_probs=(1e-8, 1e-7, 1.0))).answer(req)
    assert result.response.answers["öfke"].score == 2.0


def _sizes():
    model = FakeOpenJev()
    adapter = OpenJevAdapter(model=model)
    req = request()
    joint = len(adapter._sequence(model, req))
    singles = [len(adapter._sequence(model, _only(req, qid))) for qid in req.questions]
    return joint, max(singles)


def test_questions_that_fit_only_one_at_a_time_are_answered_one_per_pass():
    joint, single = _sizes()
    model = FakeOpenJev(limit=joint - 1)
    adapter = OpenJevAdapter(model=model)
    assert adapter.unanswerable(request()) is None
    result = adapter.answer(request())
    assert len(model.calls) == len(QUESTIONS)
    assert all(d["questions_per_pass"] == 1 for d in result.diagnostics.values())
    whole = OpenJevAdapter(model=FakeOpenJev()).answer(request())
    assert all(d["questions_per_pass"] == len(QUESTIONS) for d in whole.diagnostics.values())


def test_a_question_over_the_models_sequence_limit_alone_is_unanswerable():
    joint, single = _sizes()
    model = FakeOpenJev(limit=single - 1)
    adapter = OpenJevAdapter(model=model)
    assert "max_len" in adapter.unanswerable(request())
    assert model.calls == []
    assert OpenJevAdapter(model=FakeOpenJev()).unanswerable(request()) is None


def test_two_levels_with_the_same_text_are_unanswerable():
    questions = {"öfke": {"type": "score", "instructions": "?", "criteria": ["aynı", "aynı"]}}
    assert "same text" in OpenJevAdapter(model=FakeOpenJev()).unanswerable(request(questions))


def test_an_answer_over_other_options_stops_the_run():
    model = FakeOpenJev()
    original = model.decide
    model.decide = lambda state, qs: [
        {**a, "probabilities": {"x": 1.0}} if "probabilities" in a else a
        for a in original(state, qs)
    ]
    with pytest.raises(AdapterError, match="options"):
        OpenJevAdapter(model=model).answer(request())


def test_structured_instructions_are_sent_as_json_text():
    q = request({"a": {"type": "noul", "instructions": {"soru": "İade mi?"}}}).questions["a"]
    assert model_question(q) == {"type": "noul", "instructions": '{"soru": "İade mi?"}'}


def test_info_names_the_pinned_revision_and_the_mapping():
    info = OpenJevAdapter(model=FakeOpenJev()).info()
    assert (info.adapter, info.model, info.revision) == ("openjev", REPO, REVISION)
    assert "temperature 1.05" in info.path_used
    assert "criteria keys" in info.path_used
