"""The decision-API adapter, tested against the contract's documented bodies.

No network and no key: every test injects an httpx.MockTransport. The
request and response bodies are the examples from docs.typesafe.ai/api,
copied on 2026-09-19.
"""

import json

import httpx
import pytest

from bench.adapters.base import AdapterError
from bench.adapters.typesafe import TypeSafeAdapter
from schema.api import Request

SECRET = "sk-test-0123456789-do-not-leak"

REQUEST = Request.model_validate(
    {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "whatever-the-harness-put-here",
        "questions": {
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": None,
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    }
)

RESPONSE_BODY = {
    "model": "jev-1.13",
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


def make(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleeps = []
    adapter = TypeSafeAdapter(api_key=SECRET, client=client, sleep=sleeps.append, **kwargs)
    return adapter, sleeps


def test_the_request_follows_the_documented_contract():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=RESPONSE_BODY)

    adapter, _ = make(handler)
    adapter.answer(REQUEST)

    (sent,) = seen
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.typesafe.ai/v1/systemone"
    assert sent.headers["authorization"] == f"Bearer {SECRET}"
    assert sent.headers["content-type"].startswith("application/json")
    body = json.loads(sent.content)
    # The adapter's model replaces whatever the harness put in the request.
    assert body["model"] == "jev-latest"
    assert body["state"] == REQUEST.state
    assert body["questions"]["department"]["criteria"] == {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": None,
    }
    # Optional fields that were not set are left out, not sent as null.
    assert "criteria" not in body["questions"]["is_urgent"]
    # Our abstain extension is never sent to a server that does not know it.
    assert "abstain" not in json.dumps(body)


def test_the_response_is_parsed_and_described():
    adapter, _ = make(lambda request: httpx.Response(200, json=RESPONSE_BODY))
    result = adapter.answer(REQUEST)

    assert result.response.answers["department"].choice == "technical"
    assert result.response.answers["is_urgent"].noul == 0.92
    assert result.response.usage.input_tokens == 312
    assert result.latency_ms > 0
    # One call answers every question, so there is no per-question timing.
    assert result.per_question_ms is None
    # The API returns its own confidence on choice and score; a noul has none.
    assert result.confidence_source == {"department": "model", "frustration": "model"}
    # The alias "jev-latest" resolves to a dated model; keep what was served.
    assert result.diagnostics["department"]["served_model"] == "jev-1.13"


def test_info():
    adapter, _ = make(lambda request: httpx.Response(200, json=RESPONSE_BODY), model="jev-latest")
    info = adapter.info()
    # The model id asked for is the revision recorded; the served dated id is per row.
    assert (info.adapter, info.model, info.revision) == ("typesafe", "jev-latest", "jev-latest")
    assert info.path_used == "typesafe api"
    assert info.device is None and info.prompt_version is None


def test_base_url_and_model_can_be_changed():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=RESPONSE_BODY)

    adapter, _ = make(handler, base_url="http://localhost:9000/", model="jev-1.13")
    adapter.answer(REQUEST)
    assert str(seen[0].url) == "http://localhost:9000/v1/systemone"
    assert json.loads(seen[0].content)["model"] == "jev-1.13"


@pytest.mark.parametrize("status", [429, 529])
def test_rate_limits_are_retried_with_growing_delays(status):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(status, json={"error": "slow down"})
        return httpx.Response(200, json=RESPONSE_BODY)

    adapter, sleeps = make(handler, max_retries=3)
    result = adapter.answer(REQUEST)
    assert len(calls) == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] > 0
    assert result.response.answers["is_urgent"].noul == 0.92


def test_retries_are_bounded():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": "slow down"})

    adapter, sleeps = make(handler, max_retries=2)
    with pytest.raises(AdapterError, match="429"):
        adapter.answer(REQUEST)
    assert len(calls) == 3  # the first attempt plus two retries


@pytest.mark.parametrize("status", [401, 422])
def test_client_errors_are_not_retried_and_show_the_server_message(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"detail": "questions.department.criteria is required"})

    adapter, sleeps = make(handler)
    with pytest.raises(AdapterError) as caught:
        adapter.answer(REQUEST)
    assert len(calls) == 1 and sleeps == []
    assert str(status) in str(caught.value)
    assert "criteria is required" in str(caught.value)


def test_a_network_failure_becomes_an_adapter_error():
    def handler(request):
        raise httpx.ConnectError("no route to host")

    adapter, _ = make(handler, max_retries=0)
    with pytest.raises(AdapterError):
        adapter.answer(REQUEST)


def test_the_key_never_appears_in_an_error():
    def handler(request):
        return httpx.Response(401, json={"detail": f"bad key {SECRET}"})

    adapter, _ = make(handler)
    with pytest.raises(AdapterError) as caught:
        adapter.answer(REQUEST)
    assert SECRET not in str(caught.value)
    assert SECRET not in repr(caught.value)


def test_the_key_does_not_survive_in_a_chained_cause():
    # A 200 whose body is not the contract and echoes the key. The validation
    # error quotes its input, so it must not be chained onto the AdapterError.
    body = {"model": "m", "answers": {"is_urgent": {"type": "noul", "noul": SECRET}}}
    adapter, _ = make(lambda request: httpx.Response(200, json=body))
    with pytest.raises(AdapterError) as caught:
        adapter.answer(REQUEST)
    assert SECRET not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_a_redirect_is_not_a_success():
    adapter, sleeps = make(lambda request: httpx.Response(301, headers={"location": "/x"}))
    with pytest.raises(AdapterError, match="301"):
        adapter.answer(REQUEST)
    assert sleeps == []


def test_the_key_comes_from_the_environment_when_not_passed(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=RESPONSE_BODY)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    TypeSafeAdapter(client=client).answer(REQUEST)
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}"


def test_a_missing_key_is_reported_before_any_request(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="TYPESAFE_API_KEY"):
        TypeSafeAdapter()


def test_an_answer_set_that_does_not_match_the_questions_is_refused():
    body = {**RESPONSE_BODY, "answers": {"is_urgent": {"type": "noul", "noul": 0.5}}}
    adapter, _ = make(lambda request: httpx.Response(200, json=body))
    with pytest.raises(AdapterError, match="missing"):
        adapter.answer(REQUEST)


def test_a_body_that_is_not_the_contract_is_refused():
    adapter, _ = make(lambda request: httpx.Response(200, json={"hello": "world"}))
    with pytest.raises(AdapterError):
        adapter.answer(REQUEST)


def test_rounded_probabilities_are_rescaled_and_the_returned_sum_kept():
    body = json.loads(json.dumps(RESPONSE_BODY))
    for answer in body["answers"].values():
        if "probabilities" in answer:
            keys = list(answer["probabilities"])
            answer["probabilities"] = {
                k: (0.49 if n == 0 else 0.5 / (len(keys) - 1)) for n, k in enumerate(keys)
            }
            if "choice" in answer:
                answer["choice"] = keys[0]
    adapter, _ = make(lambda request: httpx.Response(200, json=body))
    result = adapter.answer(REQUEST)
    for qid, answer in result.response.answers.items():
        if answer.type != "noul":
            assert abs(sum(answer.probabilities.values()) - 1.0) < 1e-9
            assert result.diagnostics[qid]["probability_sum_returned"] == 0.99


def test_a_sum_far_from_one_is_still_refused():
    body = json.loads(json.dumps(RESPONSE_BODY))
    for answer in body["answers"].values():
        if "probabilities" in answer:
            answer["probabilities"] = {k: 0.1 for k in answer["probabilities"]}
    adapter, _ = make(lambda request: httpx.Response(200, json=body))
    with pytest.raises(AdapterError):
        adapter.answer(REQUEST)


def test_a_choice_moved_by_rounding_follows_the_probabilities_and_is_kept():
    body = json.loads(json.dumps(RESPONSE_BODY))
    answer = body["answers"]["department"]
    keys = list(answer["probabilities"])
    answer["probabilities"] = {k: 0.0 for k in keys}
    answer["probabilities"][keys[0]] = 0.41
    answer["probabilities"][keys[1]] = 0.40
    answer["probabilities"][keys[2]] = 0.19
    answer["choice"] = keys[1]
    adapter, _ = make(lambda request: httpx.Response(200, json=body))
    result = adapter.answer(REQUEST)
    assert result.response.answers["department"].choice == keys[0]
    assert result.diagnostics["department"]["choice_returned"] == keys[1]
