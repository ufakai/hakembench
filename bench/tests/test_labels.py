"""Prompt rendering and the label-probability arithmetic."""

import math

import pytest

from bench.adapters.base import AdapterError
from bench.adapters.labels import (
    PROMPT_VERSION,
    answer_from_label_logprobs,
    collect_label_logprobs,
    distribution,
    prompt_version,
    render,
)
from schema.questions import ChoiceQuestion, NoulQuestion, ScoreQuestion

CHOICE = ChoiceQuestion(
    instructions="Hangi birim?",
    criteria={"teknik destek": "Arızalar", "faturalandırma": None, "satış": "Kampanyalar"},
)
SCORE = ScoreQuestion(instructions="Ne kadar öfkeli?", criteria=["Sakin", "Rahatsız", "Öfkeli"])
NOUL = NoulQuestion(instructions="Acil mi?")


def logs(*probabilities):
    return [math.log(p) for p in probabilities]


def test_choice_prompt_lists_options_in_order_and_ends_at_the_answer_slot():
    labelled = render("İnternetim yok.", CHOICE)
    assert labelled.labels == ["A", "B", "C"]
    assert labelled.keys == ["teknik destek", "faturalandırma", "satış"]
    assert "A) teknik destek: Arızalar\nB) faturalandırma\nC) satış: Kampanyalar" in labelled.prompt
    assert labelled.prompt.startswith("Metin:\nİnternetim yok.\n\nSoru: Hangi birim?\n")
    assert labelled.prompt.endswith("Cevap:")
    assert labelled.chat_prompt.endswith("Cevap:")
    assert "harfini" in labelled.chat_prompt and "harfini" not in labelled.prompt


def test_score_and_noul_keys():
    assert render("s", SCORE).keys == ["0", "1", "2"]
    noul = render("s", NOUL)
    assert noul.keys == ["true", "false"]
    assert "A) Evet\nB) Hayır" in noul.prompt


def test_the_turkish_frame_is_byte_identical_to_labels_v1():
    """Rows made before the English frame existed stay comparable with new Turkish rows."""
    options = "A) teknik destek: Arızalar\nB) faturalandırma\nC) satış: Kampanyalar\n"
    head = "Metin:\nİnternetim yok.\n\nSoru: Hangi birim?\nSeçenekler:\n"
    for labelled in (render("İnternetim yok.", CHOICE), render("İnternetim yok.", CHOICE, "tr")):
        assert labelled.prompt == head + options + "Cevap:"
        assert labelled.chat_prompt == (
            head + options + "Yalnızca doğru seçeneğin harfini yaz.\nCevap:"
        )
    noul = render("İnternetim yok.", NOUL, "tr")
    assert noul.chat_prompt == (
        "Metin:\nİnternetim yok.\n\nSoru: Acil mi?\nSeçenekler:\nA) Evet\nB) Hayır\n"
        "Yalnızca doğru seçeneğin harfini yaz.\nCevap:"
    )
    assert prompt_version() == prompt_version("tr") == PROMPT_VERSION == "labels-v1"


def test_the_english_frame_keeps_letters_and_keys():
    criteria = {"support": "Faults", "sales": None}
    question = ChoiceQuestion(instructions="Which unit?", criteria=criteria)
    labelled = render("No internet.", question, "en")
    assert labelled.prompt == (
        "Text:\nNo internet.\n\nQuestion: Which unit?\nOptions:\nA) support: Faults\nB) sales\n"
        "Answer:"
    )
    assert labelled.chat_prompt.endswith("Write only the letter of the correct option.\nAnswer:")
    assert labelled.labels == ["A", "B"] and labelled.keys == ["support", "sales"]
    noul = render("s", NoulQuestion(instructions="Urgent?"), "en")
    assert "A) Yes\nB) No" in noul.prompt and noul.keys == ["true", "false"]
    assert render("s", SCORE, "en").keys == ["0", "1", "2"]
    assert prompt_version("en") == "labels-v1+en"
    with pytest.raises(ValueError, match="prompt language"):
        render("s", NOUL, "de")
    with pytest.raises(ValueError, match="prompt language"):
        prompt_version("de")


def test_structured_state_is_rendered_as_readable_json():
    labelled = render({"mesaj": "İade istiyorum", "sipariş": 104}, NOUL)
    assert '"mesaj": "İade istiyorum"' in labelled.prompt


def test_more_options_than_letters_is_refused():
    big = ChoiceQuestion(instructions="?", criteria={f"o{i}": None for i in range(27)})
    with pytest.raises(AdapterError, match="at most 26"):
        render("s", big)


def test_token_variants_of_one_letter_are_summed():
    candidates = [(" A", math.log(0.3)), ("A", math.log(0.1)), ("B", math.log(0.2)), ("Ab", -1.0)]
    got = collect_label_logprobs(candidates, ["A", "B", "C"])
    assert got["A"] == pytest.approx(math.log(0.4))
    assert got["B"] == pytest.approx(math.log(0.2))
    assert got["C"] is None


def test_distribution_renormalises_and_reports_mass_and_missing_labels():
    probabilities, diagnostics = distribution({"A": math.log(0.4), "B": math.log(0.1), "C": None})
    assert probabilities == pytest.approx([0.8, 0.2, 0.0])
    assert math.fsum(probabilities) == pytest.approx(1.0, abs=1e-12)
    assert diagnostics["label_mass"] == pytest.approx(0.5)
    assert diagnostics["missing_labels"] == ["C"]


def test_distribution_is_stable_for_very_negative_logprobs():
    probabilities, _ = distribution({"A": -1000.0, "B": -1001.0})
    assert probabilities == pytest.approx([1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))])


def test_no_label_at_all_is_an_error():
    with pytest.raises(AdapterError):
        distribution({"A": None, "B": None})


def test_choice_answer():
    labelled = render("s", CHOICE)
    answer, _ = answer_from_label_logprobs(
        CHOICE, labelled, dict(zip("ABC", logs(0.2, 0.7, 0.1), strict=True))
    )
    assert answer.choice == "faturalandırma"
    assert answer.probabilities["faturalandırma"] == pytest.approx(0.7)
    assert answer.confidence == pytest.approx(0.7)
    assert answer.abstain is None


def test_score_answer_reports_the_expected_level():
    labelled = render("s", SCORE)
    answer, _ = answer_from_label_logprobs(
        SCORE, labelled, dict(zip("ABC", logs(0.05, 0.3, 0.65), strict=True))
    )
    assert answer.score == pytest.approx(1.6)
    assert answer.legend == {"0": "Sakin", "1": "Rahatsız", "2": "Öfkeli"}
    assert answer.argmax_level == 2


def test_noul_answer_is_the_probability_of_yes():
    labelled = render("s", NOUL)
    answer, _ = answer_from_label_logprobs(
        NOUL, labelled, dict(zip("AB", logs(0.23, 0.02), strict=True))
    )
    assert answer.noul == pytest.approx(0.92)


def test_labels_out_of_order_are_refused():
    labelled = render("s", NOUL)
    with pytest.raises(AdapterError):
        answer_from_label_logprobs(NOUL, labelled, {"B": -1.0, "A": -2.0})
