"""The schema accepts the public contract's own examples and rejects malformed input.

The JSON bodies below are the example request and response bodies from
docs.typesafe.ai/api, copied on 2026-09-19.
"""

import pytest
from pydantic import ValidationError

from schema.api import Request, Response, check_response
from schema.questions import (
    HEAD_OPTIONS_PER_PASS,
    MAX_CHOICE_OPTIONS,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
    expected_level,
)

STATE = "Help! My payouts have been failing for 3 days."

DOC_REQUEST = {
    "state": STATE,
    "model": "jev-latest",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}

DOC_RESPONSE = {
    "model": "jev-latest",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.92},
        "department": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
            "confidence": 0.82,
        },
        "frustration": {
            "type": "score",
            "score": 1.6,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
            "confidence": 0.78,
        },
    },
    "usage": {"input_tokens": 312, "output_tokens": 48},
}


def test_contract_examples_validate_and_match():
    request = Request.model_validate(DOC_REQUEST)
    response = Response.model_validate(DOC_RESPONSE)
    check_response(request, response)
    assert response.answers["department"].abstain is None


def test_request_round_trips_to_the_same_json():
    request = Request.model_validate(DOC_REQUEST)
    assert request.model_dump(mode="json", exclude_none=True) == DOC_REQUEST


def test_state_and_instructions_may_be_structured():
    body = {
        "state": {"messages": [{"from": "customer", "text": "İade istiyorum."}]},
        "model": "m",
        "questions": {"q": {"type": "noul", "instructions": ["Müşteri iade mi istiyor?"]}},
    }
    Request.model_validate(body)


def test_choice_accepts_the_api_limit_and_rejects_one_more():
    at_limit = {f"seçenek {i}": None for i in range(MAX_CHOICE_OPTIONS)}
    ChoiceQuestion(instructions="Hangisi?", criteria=at_limit)
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="Hangisi?", criteria={**at_limit, "fazla": None})


def test_api_limit_is_larger_than_the_per_pass_limit():
    # The API takes 255 options, the head scores 10 per pass, larger sets are
    # chunked inside the server. A question with 11 options is valid input.
    assert HEAD_OPTIONS_PER_PASS < MAX_CHOICE_OPTIONS
    eleven = {f"o{i}": None for i in range(HEAD_OPTIONS_PER_PASS + 1)}
    ChoiceQuestion(instructions="Hangisi?", criteria=eleven)


@pytest.mark.parametrize("criteria", [{}, {"tek": None}, {"a": None, " ": None}])
def test_choice_rejects_degenerate_option_sets(criteria):
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="Hangisi?", criteria=criteria)


@pytest.mark.parametrize("levels", [1, 11])
def test_score_rejects_level_counts_outside_two_to_ten(levels):
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="Ne kadar?", criteria=[f"düzey {i}" for i in range(levels)])


def test_unknown_question_type_is_rejected():
    body = {"state": "s", "model": "m", "questions": {"q": {"type": "rank", "instructions": "?"}}}
    with pytest.raises(ValidationError):
        Request.model_validate(body)


def test_request_needs_a_question():
    with pytest.raises(ValidationError):
        Request.model_validate({"state": "s", "model": "m", "questions": {}})


def test_choice_answer_rejects_probabilities_that_do_not_sum_to_one():
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="a", probabilities={"a": 0.6, "b": 0.3}, confidence=0.5)


def test_choice_answer_rejects_a_choice_that_is_not_the_most_probable():
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="b", probabilities={"a": 0.6, "b": 0.4}, confidence=0.5)


def test_choice_answer_allows_ties():
    ChoiceAnswer(choice="b", probabilities={"a": 0.5, "b": 0.5}, confidence=0.0)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_probabilities_outside_the_unit_interval_are_rejected(value):
    with pytest.raises(ValidationError):
        NoulAnswer(noul=value)
    with pytest.raises(ValidationError):
        NoulAnswer(noul=0.5, abstain=value)


def test_score_answer_checks_legend_and_range():
    good = DOC_RESPONSE["answers"]["frustration"]
    answer = ScoreAnswer.model_validate(good)
    assert answer.argmax_level == 2
    assert expected_level(answer.probabilities) == pytest.approx(answer.score)
    with pytest.raises(ValidationError):
        ScoreAnswer.model_validate({**good, "legend": {"1": "a", "2": "b", "3": "c"}})
    with pytest.raises(ValidationError):
        ScoreAnswer.model_validate({**good, "score": 2.5})


def test_abstain_is_carried_when_present():
    answer = NoulAnswer(noul=0.5, abstain=0.9)
    assert NoulAnswer.model_validate(answer.model_dump()).abstain == 0.9


def _response_with(**answers):
    merged = {**DOC_RESPONSE["answers"], **answers}
    return Response.model_validate({"model": "m", "answers": merged})


def test_check_response_catches_mismatches():
    request = Request.model_validate(DOC_REQUEST)

    missing = Response.model_validate({"model": "m", "answers": {}})
    with pytest.raises(ValueError, match="missing"):
        check_response(request, missing)

    wrong_type = _response_with(is_urgent=DOC_RESPONSE["answers"]["department"])
    with pytest.raises(ValueError, match="asked a noul"):
        check_response(request, wrong_type)

    wrong_options = _response_with(
        department={
            "type": "choice",
            "choice": "technical",
            "probabilities": {"technical": 0.9, "legal": 0.1},
            "confidence": 0.8,
        }
    )
    with pytest.raises(ValueError, match="options"):
        check_response(request, wrong_options)

    wrong_legend = _response_with(
        frustration={
            **DOC_RESPONSE["answers"]["frustration"],
            "legend": {"0": "Calm", "1": "Annoyed", "2": "Very angry"},
        }
    )
    with pytest.raises(ValueError, match="legend"):
        check_response(request, wrong_legend)
