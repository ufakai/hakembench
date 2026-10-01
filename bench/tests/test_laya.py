"""The Laya adapter on a fake Agent: what is sent, what comes back, what is refused. No network."""

import numpy as np
import pytest

from bench.adapters.base import AdapterError
from bench.adapters.laya import CHECKPOINTS, PACKAGE_VERSION, LayaAdapter
from schema.api import Request, check_response

STATE = "Faturam iki kez kesildi, iadeyi bugün istiyorum yoksa aboneliği iptal ederim."
QUESTIONS = {
    "birim": {"type": "choice", "instructions": "Hangi birim ilgilenmeli?",
              "criteria": {"fatura": "ödeme ve iade", "teknik": None, "satış": "yeni sözleşme"}},
    "aciliyet": {"type": "score", "instructions": "Ne kadar acil?",
                 "criteria": ["acil değil", "yakında", "hemen"]},
    "iptal": {"type": "noul", "instructions": "Aboneliği iptal etmekle tehdit ediyor mu?",
              "criteria": {"true": "evet, iptal diyor", "false": "hayır"}},
}  # fmt: skip
CLS, SEP, MASK = 1, 2, 3


class Tok:
    mask_token = "[MASK]"
    sep_token_id = SEP

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7] * len(text.split())}


class Logits:
    """Stands in for the model's logits tensor in the forward hook."""

    def __init__(self, rows):
        self.rows = rows

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.rows, dtype=np.float32)


class Model:
    def register_forward_hook(self, hook):
        self.hook = hook


class FakeAgent:
    """The parts of laya.agent.Agent the adapter touches, with the package's answer shape."""

    def __init__(self, max_options=10, state_room=6):
        self.cfg = {"max_len": 64, "head_max_len": 16, "temperature_by_options": {"noul:2": 2.0}}
        self.tok = Tok()
        self.model = Model()
        self.max_options = max_options
        self.state_room = state_room
        self.calls = []

    def _check_question(self, qid, q):
        if q["type"] not in ("choice", "score", "noul"):
            raise ValueError(f"question {qid!r}: unknown type")

    def _to_internal(self, q):
        return q

    def _encode_state(self, state, ids, internal, max_len=None, head_max_len=None):
        self.budgets = (max_len, head_max_len)
        items = []
        state_ids = self.tok(state)["input_ids"][: self.state_room]
        for qid in ids:
            q = internal[qid]
            k = 2 if q["type"] == "noul" else len(q["criteria"])
            if k > self.max_options:
                raise ValueError(f"question {qid!r} options exceed head_max_len=16")
            seq = [CLS, 5, 5, SEP]
            markers = []
            for _ in range(k):
                markers.append(len(seq))
                seq += [MASK, 6]
            seq += [SEP, *state_ids, SEP]
            items.append({"ids": seq, "markers": markers})
        return items

    def predict(self, state, questions, max_len=None, head_max_len=None):
        self.calls.append((state, questions))
        self.predict_budgets = (max_len, head_max_len)
        rows = []
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                p = np.linspace(1, 2, len(keys))
                p = p / p.sum()
                rows.append(list(np.log(p)) + [0.0] * (3 - len(keys)))
                rounded = {k: round(float(v), 4) for k, v in zip(keys, p, strict=True)}
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": rounded,
                                "confidence": 0.0123, "answer_confidence": round(float(p.max()), 4),
                                "action": {"act_probability": 1.0}}  # fmt: skip
            elif q["type"] == "score":
                p = np.array([0.2, 0.3, 0.5])
                rows.append(list(np.log(p)))
                answers[qid] = {"type": "score", "score": round(float((np.arange(3) * p).sum()), 4),
                                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                "probabilities": {str(i): float(v) for i, v in enumerate(p)},
                                "confidence": 0.05, "answer_confidence": 0.5,
                                "action": {"act_probability": 0.9}}  # fmt: skip
            else:
                rows.append([0.0, 1.0, 0.0])
                answers[qid] = {"type": "noul", "noul": 0.7311, "confidence": 0.7311,
                                "answer_confidence": 0.7311,
                                "action": {"act_probability": 1.0}}  # fmt: skip
        if hasattr(self.model, "hook"):
            self.model.hook(self.model, (), (Logits(rows), None))
        return {"model": "laya-rl-agent", "answers": answers,
                "usage": {"input_tokens": 40, "output_tokens": 0}}  # fmt: skip


def request(questions=QUESTIONS) -> Request:
    return Request.model_validate({"state": STATE, "model": "x", "questions": questions})


def test_answers_are_the_packages_own_in_the_contracts_shape():
    agent = FakeAgent()
    adapter = LayaAdapter("laya", agent=agent)
    req = request()
    result = adapter.answer(req)
    check_response(req, result.response)

    birim = result.response.answers["birim"]
    assert birim.confidence == 0.0123  # the package's entropy confidence, unchanged
    assert birim.choice == "satış"
    assert result.response.answers["iptal"].noul == 0.7311
    assert result.response.answers["aciliyet"].score == pytest.approx(1.3)
    # The package's own confidence on choice and score; a noul has none.
    assert result.confidence_source == {"birim": "model", "aciliyet": "model"}
    assert result.per_question_ms is None  # one call answers the request
    assert result.diagnostics["birim"]["answer_confidence"] == pytest.approx(0.4444, abs=1e-4)
    assert result.diagnostics["aciliyet"]["act_probability"] == 0.9


def test_the_state_and_questions_go_in_as_the_benchmark_has_them():
    agent = FakeAgent()
    LayaAdapter("laya-multilingual", agent=agent).answer(request())
    (state, sent), = agent.calls  # fmt: skip
    assert state == STATE
    assert sent["birim"]["criteria"] == QUESTIONS["birim"]["criteria"]  # a null description stays
    assert sent["aciliyet"] == QUESTIONS["aciliyet"]
    assert sent["iptal"]["criteria"] == {"true": "evet, iptal diyor", "false": "hayır"}
    # Nothing added: no labels, no extra instruction text.
    assert all(set(q) <= {"type", "instructions", "criteria"} for q in sent.values())


def test_noul_without_criteria_sends_none():
    agent = FakeAgent()
    questions = {"iptal": {"type": "noul", "instructions": "İptal mi?"}}
    LayaAdapter("laya", agent=agent).answer(request(questions))
    assert agent.calls[0][1]["iptal"] == {"type": "noul", "instructions": "İptal mi?"}


def test_diagnostics_keep_raw_logits_and_how_much_state_was_read():
    agent = FakeAgent(state_room=4)
    result = LayaAdapter("laya", agent=agent).answer(request())
    birim = result.diagnostics["birim"]
    assert len(birim["logits"]) == 3
    assert birim["state_tokens"] == len(STATE.split())
    assert birim["state_tokens_read"] == 4
    assert len(result.diagnostics["iptal"]["logits"]) == 2


def test_an_item_the_package_cannot_encode_is_unanswerable_before_any_forward_pass():
    agent = FakeAgent(max_options=2)
    adapter = LayaAdapter("laya", agent=agent)
    reason = adapter.unanswerable(request())
    assert "exceed head_max_len" in reason
    assert agent.calls == []
    assert LayaAdapter("laya", agent=FakeAgent()).unanswerable(request()) is None


def test_a_wrong_answer_type_stops_the_run():
    agent = FakeAgent()
    original = agent.predict

    def swapped(state, questions, **budget):
        out = original(state, questions, **budget)
        out["answers"]["iptal"] = out["answers"]["aciliyet"]
        return out

    agent.predict = swapped
    with pytest.raises(AdapterError, match="noul"):
        LayaAdapter("laya", agent=agent).answer(request())


def test_info_names_the_pinned_checkpoint_package_and_route():
    info = LayaAdapter("laya", agent=FakeAgent()).info()
    assert info.adapter == "laya"
    assert info.model == CHECKPOINTS["laya"].repo
    assert info.revision == CHECKPOINTS["laya"].revision
    assert f"laya {PACKAGE_VERSION}" in info.path_used
    assert "no Router" in info.path_used
    assert "per question type and option count" in info.path_used
    assert info.device == "cpu"


def test_unknown_checkpoint_is_refused():
    with pytest.raises(AdapterError, match="unknown"):
        LayaAdapter("laya-typed-decisions", agent=FakeAgent())


def test_the_packages_per_call_budget_override_reaches_both_calls_and_the_rows():
    agent = FakeAgent()
    adapter = LayaAdapter("laya", agent=agent, max_len=4096, head_max_len=1024)
    adapter.answer(request())
    assert agent.budgets == (4096, 1024) and agent.predict_budgets == (4096, 1024)
    shipped = FakeAgent()
    LayaAdapter("laya", agent=shipped).answer(request())
    assert shipped.budgets == (None, None) and shipped.predict_budgets == (None, None)
    from bench.adapters.laya import _budget_text

    cfg = {"max_len": 512, "head_max_len": 192}
    assert "per-call override" in _budget_text(cfg, 4096, 1024)
    assert "as shipped" in _budget_text(cfg, None, None)
