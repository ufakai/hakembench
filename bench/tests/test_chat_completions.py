"""The chat-completions adapter: label reading over a hosted or local endpoint.

No network: every test injects an httpx.MockTransport that answers in the
chat-completions response format with token log-probabilities.
"""

import json
import math

import httpx
import pytest

from bench.adapters.base import AdapterError
from bench.adapters.chat_completions import ChatCompletionsAdapter
from bench.adapters.labels import PROMPT_VERSION, render
from schema.api import Request

SECRET = "sk-test-0123456789-do-not-leak"

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
            "acil": {"type": "noul", "instructions": "Acil mi?"},
        },
    }
)


def completion(top, prompt_tokens=50):
    """A chat-completions body whose first generated token has these top log-probabilities."""
    top_logprobs = [{"token": token, "logprob": math.log(p), "bytes": None} for token, p in top]
    first = {"token": top[0][0], "logprob": math.log(top[0][1]), "top_logprobs": top_logprobs}
    return {
        "id": "x",
        "model": "served-model-name",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": top[0][0]},
                "logprobs": {"content": [first]},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1},
    }


# Served in order: the choice question first, then the noul.
BODIES = [
    completion([("A", 0.6), ("B", 0.15), (" A", 0.1), ("C", 0.05), ("Merhaba", 0.1)]),
    completion([("B", 0.72), ("A", 0.18), ("Evet", 0.1)], prompt_tokens=40),
]


def make(bodies=BODIES, **kwargs):
    seen = []
    queue = list(bodies)

    def handler(request):
        seen.append(request)
        body = queue.pop(0)
        if isinstance(body, httpx.Response):
            return body
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs.setdefault("base_url", "http://localhost:8080/v1")
    adapter = ChatCompletionsAdapter(model="ufakzeka-1", client=client, **kwargs)
    return adapter, seen


def test_one_call_per_question_with_the_shared_prompt():
    adapter, seen = make()
    adapter.answer(REQUEST)

    assert [str(r.url) for r in seen] == ["http://localhost:8080/v1/chat/completions"] * 2
    body = json.loads(seen[0].content)
    labelled = render(REQUEST.state, REQUEST.questions["birim"])
    assert body["model"] == "ufakzeka-1"
    assert body["messages"] == [{"role": "user", "content": labelled.chat_prompt}]
    # One token, greedy, with the top log-probabilities of that token.
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    assert body.get("stream", False) is False


def test_probabilities_are_renormalised_over_the_labels():
    adapter, _ = make()
    result = adapter.answer(REQUEST)

    choice = result.response.answers["birim"]
    # "A" and " A" are the same label: 0.6 + 0.1. Mass on labels: 0.9.
    assert choice.probabilities["teknik destek"] == pytest.approx(0.7 / 0.9)
    assert choice.probabilities["faturalandırma"] == pytest.approx(0.15 / 0.9)
    assert choice.probabilities["satış"] == pytest.approx(0.05 / 0.9)
    assert choice.choice == "teknik destek"
    assert choice.confidence == pytest.approx(0.7 / 0.9)
    assert choice.abstain is None

    noul = result.response.answers["acil"]
    # A is yes, B is no.
    assert noul.noul == pytest.approx(0.18 / 0.9)


def test_the_result_is_described():
    adapter, _ = make()
    result = adapter.answer(REQUEST)

    assert result.confidence_source == {"birim": "max_probability"}
    assert set(result.per_question_ms) == {"birim", "acil"}
    assert all(ms > 0 for ms in result.per_question_ms.values())
    assert result.latency_ms >= sum(result.per_question_ms.values()) * 0.99
    assert result.diagnostics["birim"]["label_mass"] == pytest.approx(0.9)
    assert result.diagnostics["birim"]["missing_labels"] == []
    assert result.diagnostics["birim"]["served_model"] == "served-model-name"
    # Token counts are summed over the calls.
    assert result.response.usage.input_tokens == 90
    assert result.response.usage.output_tokens == 2
    assert result.response.model == "ufakzeka-1"


def test_a_label_outside_the_top_tokens_gets_zero_and_is_reported():
    bodies = [completion([("A", 0.9), ("B", 0.1)]), BODIES[1]]
    adapter, _ = make(bodies)
    result = adapter.answer(REQUEST)
    assert result.response.answers["birim"].probabilities["satış"] == 0.0
    assert result.diagnostics["birim"]["missing_labels"] == ["C"]


def test_info_and_the_route_description():
    adapter, _ = make()
    info = adapter.info()
    assert (info.adapter, info.model) == ("chat_completions", "ufakzeka-1")
    assert info.prompt_version == PROMPT_VERSION
    assert info.path_used == "chat-completions endpoint"
    assert info.revision is None

    described, _ = make(path_used="GGUF q8_0 through llama-server", revision="sha256:abc")
    assert described.info().path_used == "GGUF q8_0 through llama-server"
    assert described.info().revision == "sha256:abc"


def test_no_key_means_no_authorization_header():
    adapter, seen = make()
    adapter.answer(REQUEST)
    assert "authorization" not in seen[0].headers


def test_a_key_is_sent_as_a_bearer_token_and_never_leaks():
    failing = [httpx.Response(401, json={"error": {"message": f"invalid key {SECRET}"}})]
    adapter, seen = make(failing, api_key=SECRET, base_url="https://example.invalid/api/v1/")
    with pytest.raises(AdapterError) as caught:
        adapter.answer(REQUEST)
    assert str(seen[0].url) == "https://example.invalid/api/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}"
    assert "401" in str(caught.value)
    assert SECRET not in str(caught.value) and SECRET not in repr(caught.value)


def test_a_server_that_returns_no_logprobs_is_reported_clearly():
    body = completion([("A", 0.9)])
    body["choices"][0]["logprobs"] = None
    adapter, _ = make([body])
    with pytest.raises(AdapterError, match="log-probabilities"):
        adapter.answer(REQUEST)


@pytest.mark.parametrize(
    "top_logprobs",
    [[{"logprob": -0.1}], [{"token": "A", "logprob": "high"}], ["A"], "A"],
)
def test_malformed_log_probabilities_are_an_adapter_error(top_logprobs):
    body = completion([("A", 0.9)])
    body["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = top_logprobs
    adapter, _ = make([body])
    with pytest.raises(AdapterError, match="format"):
        adapter.answer(REQUEST)


def test_no_label_among_the_top_tokens_is_an_error():
    adapter, _ = make([completion([("Merhaba", 0.9), ("Tabii", 0.1)])])
    with pytest.raises(AdapterError):
        adapter.answer(REQUEST)


def test_a_network_failure_becomes_an_adapter_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = ChatCompletionsAdapter(model="m", base_url="http://localhost:8080/v1", client=client)
    with pytest.raises(AdapterError, match="localhost:8080"):
        adapter.answer(REQUEST)


def test_top_logprobs_is_capped_at_twenty():
    with pytest.raises(ValueError):
        ChatCompletionsAdapter(model="m", base_url="http://localhost:8080/v1", top_logprobs=21)


def test_the_english_frame_is_sent_and_recorded():
    adapter, seen = make(prompt_lang="en")
    adapter.answer(REQUEST)
    body = json.loads(seen[0].content)
    labelled = render(REQUEST.state, REQUEST.questions["birim"], "en")
    assert body["messages"] == [{"role": "user", "content": labelled.chat_prompt}]
    assert labelled.chat_prompt.startswith("Text:\n") and "Answer:" in labelled.chat_prompt
    assert adapter.info().prompt_version == PROMPT_VERSION + "+en"
    assert make()[0].info().prompt_version == PROMPT_VERSION
    with pytest.raises(ValueError, match="prompt language"):
        make(prompt_lang="de")


def test_extra_fields_are_sent_and_reasoning_tokens_kept():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        body = BODIES[len(seen) - 1] if len(seen) <= len(BODIES) else BODIES[-1]
        usage = {**body["usage"], "completion_tokens_details": {"reasoning_tokens": 7}}
        body = {**body, "usage": usage}
        return httpx.Response(200, json=body)

    adapter = ChatCompletionsAdapter(
        model="m", base_url="https://api.test/v1", api_key=SECRET,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        extra={"reasoning_effort": "low", "seed": 7}, max_tokens=400,
    )  # fmt: skip
    result = adapter.answer(REQUEST)
    assert seen[0]["reasoning_effort"] == "low" and seen[0]["seed"] == 7
    assert seen[0]["max_tokens"] == 400
    assert result.diagnostics["birim"]["reasoning_tokens"] == 7


def test_log_probabilities_off_the_visible_answer_are_refused():
    body = completion([("A", 0.6), ("B", 0.4)])
    body["choices"][0]["message"]["content"] = "B"

    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    adapter = ChatCompletionsAdapter(
        model="m", base_url="https://api.test/v1", client=httpx.Client(transport=transport)
    )
    with pytest.raises(AdapterError, match="visible answer"):
        adapter.answer(REQUEST)
    body = completion([("Düşünüyorum", 0.6), ("A", 0.4)])
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    adapter = ChatCompletionsAdapter(
        model="m", base_url="https://api.test/v1", client=httpx.Client(transport=transport)
    )
    with pytest.raises(AdapterError, match="not an answer label"):
        adapter.answer(REQUEST)
