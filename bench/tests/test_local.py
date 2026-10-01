"""The local adapter: label reading from weights on this machine.

The unit tests inject a fake tokenizer and a fake model, so they need torch
but no download. Run them where torch is installed. The one test that loads
real weights runs only when KARAR_RUN_LOCAL_MODEL=1.
"""

import math
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from bench.adapters.base import AdapterError  # noqa: E402
from bench.adapters.labels import PROMPT_VERSION, render  # noqa: E402
from bench.adapters.local import LocalAdapter  # noqa: E402
from schema.api import Request  # noqa: E402

VOCAB = 400
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

REQUEST = Request.model_validate(
    {
        "state": "Üç gündür internetim yok.",
        "model": "ignored",
        "questions": {
            "birim": {
                "type": "choice",
                "instructions": "Hangi birim?",
                "criteria": {"teknik destek": "Arızalar", "faturalandırma": None, "satış": None},
            },
            "öfke": {
                "type": "score",
                "instructions": "Ne kadar öfkeli?",
                "criteria": ["Sakin", "Rahatsız", "Öfkeli"],
            },
        },
    }
)


class FakeTokenizer:
    """Ids 0..25 are " A".." Z", ids 26..51 are "A".."Z". Other text is one id per character."""

    def __init__(self, spaced_single=True):
        self.spaced_single = spaced_single
        self.prompts = []

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if len(text) == 2 and text[0] == " " and text[1] in LETTERS and self.spaced_single:
            return [LETTERS.index(text[1])]
        if len(text) == 1 and text in LETTERS:
            return [26 + LETTERS.index(text)]
        return [100 + (ord(ch) % 200) for ch in text]

    def __call__(self, text, return_tensors=None):
        assert return_tensors == "pt"
        self.prompts.append(text)
        ids = torch.tensor([self.encode(text)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeModel(torch.nn.Module):
    """Puts the given logits on the given token ids at the last position."""

    def __init__(self, last_logits, max_positions=4096, commit="abc123"):
        super().__init__()
        self.last_logits = last_logits
        self.config = SimpleNamespace(max_position_embeddings=max_positions, _commit_hash=commit)
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.calls += 1
        logits = torch.full((1, input_ids.shape[1], VOCAB), -20.0)
        # A decoy on an earlier position: only the last position may be read.
        logits[0, 0, 0] = 50.0
        for token_id, value in self.last_logits.items():
            logits[0, -1, token_id] = value
        return SimpleNamespace(logits=logits)


def adapter_with(last_logits, **kwargs):
    tokenizer = kwargs.pop("tokenizer", None) or FakeTokenizer()
    model = FakeModel(last_logits, **{k: kwargs.pop(k) for k in ("max_positions",) if k in kwargs})
    return LocalAdapter(model_id="fake/model", tokenizer=tokenizer, model=model, **kwargs), model


def softmax(values):
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    return [e / sum(exps) for e in exps]


def test_probabilities_are_the_softmax_over_the_label_tokens():
    # " A", " B", " C" are ids 0, 1, 2. Id 300 is a non-label token with a large logit.
    adapter, model = adapter_with({0: 2.0, 1: 1.0, 2: -1.0, 300: 5.0})
    result = adapter.answer(REQUEST)

    expected = softmax([2.0, 1.0, -1.0])
    choice = result.response.answers["birim"]
    assert list(choice.probabilities) == ["teknik destek", "faturalandırma", "satış"]
    assert list(choice.probabilities.values()) == pytest.approx(expected, rel=1e-5)
    assert choice.choice == "teknik destek"
    assert choice.confidence == pytest.approx(expected[0], rel=1e-5)

    score = result.response.answers["öfke"]
    assert score.score == pytest.approx(expected[1] + 2 * expected[2], rel=1e-5)
    # One forward pass per question, nothing generated.
    assert model.calls == 2


def test_label_mass_is_measured_against_the_full_vocabulary():
    adapter, _ = adapter_with({0: 2.0, 1: 1.0, 2: -1.0, 300: 5.0})
    result = adapter.answer(REQUEST)
    full = softmax([2.0, 1.0, -1.0, 5.0] + [-20.0] * (VOCAB - 4))
    assert result.diagnostics["birim"]["label_mass"] == pytest.approx(sum(full[:3]), rel=1e-4)
    assert result.diagnostics["birim"]["missing_labels"] == []
    assert result.diagnostics["birim"]["label_tokens"] == [" A", " B", " C"]


def test_the_model_sees_the_base_prompt_not_the_chat_prompt():
    tokenizer = FakeTokenizer()
    adapter, _ = adapter_with({0: 1.0, 1: 0.0, 2: 0.0}, tokenizer=tokenizer)
    adapter.answer(REQUEST)
    assert tokenizer.prompts[0] == render(REQUEST.state, REQUEST.questions["birim"]).prompt
    assert tokenizer.prompts[0].endswith("Cevap:")


def test_all_labels_fall_back_to_the_bare_letter_together():
    # No " A" style token is a single token here, so every label uses "A" style: ids 26, 27, 28.
    tokenizer = FakeTokenizer(spaced_single=False)
    adapter, _ = adapter_with({26: 0.0, 27: 3.0, 28: 0.0, 0: 9.0}, tokenizer=tokenizer)
    result = adapter.answer(REQUEST)
    assert result.response.answers["birim"].choice == "faturalandırma"
    assert result.diagnostics["birim"]["label_tokens"] == ["A", "B", "C"]


def test_a_tokenizer_without_single_token_labels_is_refused():
    class NoLabels(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [100, 101]

    adapter, _ = adapter_with({0: 1.0}, tokenizer=NoLabels())
    with pytest.raises(AdapterError, match="single token"):
        adapter.answer(REQUEST)


def test_a_tokenizer_that_appends_an_end_token_is_refused():
    class AppendsEnd(FakeTokenizer):
        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([self.encode(text) + [399]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    adapter, model = adapter_with({0: 1.0, 1: 0.0, 2: 0.0}, tokenizer=AppendsEnd())
    with pytest.raises(AdapterError, match="answer slot"):
        adapter.answer(REQUEST)
    assert model.calls == 0


def test_a_tokenizer_that_prepends_a_start_token_is_fine():
    class PrependsStart(FakeTokenizer):
        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([[398] + self.encode(text)])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    adapter, _ = adapter_with({0: 3.0, 1: 0.0, 2: 0.0}, tokenizer=PrependsStart())
    assert adapter.answer(REQUEST).response.answers["birim"].choice == "teknik destek"


def test_a_prompt_longer_than_the_context_is_refused_not_truncated():
    adapter, model = adapter_with({0: 1.0, 1: 0.0, 2: 0.0}, max_positions=16)
    with pytest.raises(AdapterError, match="context"):
        adapter.answer(REQUEST)
    assert model.calls == 0


def test_the_result_and_the_model_are_described():
    adapter, _ = adapter_with({0: 1.0, 1: 0.0, 2: 0.0})
    result = adapter.answer(REQUEST)
    assert result.confidence_source == {"birim": "max_probability", "öfke": "max_probability"}
    assert set(result.per_question_ms) == {"birim", "öfke"}
    assert all(ms > 0 for ms in result.per_question_ms.values())
    assert result.response.model == "fake/model"
    assert result.response.usage.output_tokens == 0
    assert result.response.usage.input_tokens > 0

    info = adapter.info()
    assert (info.adapter, info.model, info.revision) == ("local", "fake/model", "abc123")
    assert info.path_used == "local adapter, Hugging Face weights"
    assert info.device == "cpu"
    assert info.prompt_version == PROMPT_VERSION


def test_gradients_are_off_and_the_model_is_in_eval_mode():
    seen = {}

    class Watching(FakeModel):
        def forward(self, input_ids, attention_mask=None, **kwargs):
            seen["grad"] = torch.is_grad_enabled()
            seen["training"] = self.training
            return super().forward(input_ids, attention_mask)

    model = Watching({0: 1.0, 1: 0.0, 2: 0.0})
    model.train()
    LocalAdapter(model_id="fake/model", tokenizer=FakeTokenizer(), model=model).answer(REQUEST)
    assert seen == {"grad": False, "training": False}


@pytest.mark.local_model
@pytest.mark.skipif(
    os.environ.get("KARAR_RUN_LOCAL_MODEL") != "1",
    reason="downloads weights; set KARAR_RUN_LOCAL_MODEL=1 to run",
)
def test_the_real_base_model_answers_a_question():
    adapter = LocalAdapter(model_id="ufakai/ufakzeka-1-base")
    result = adapter.answer(REQUEST)
    probabilities = result.response.answers["birim"].probabilities
    assert math.fsum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    # The tokenizer check from the research notes: " A".." Z" are single tokens.
    assert result.diagnostics["birim"]["label_tokens"] == [" A", " B", " C"]
    assert adapter.info().revision
