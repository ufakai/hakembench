"""The stated-probabilities adapter: one request per item, a strict schema, rescaled sums."""

import json

import httpx
import pytest

from bench.adapters.base import AdapterError
from bench.adapters.stated import PROMPT_VERSION, StatedAdapter
from schema.api import Request

REQUEST = Request.model_validate({
    "state": "Kargom üç gündür gelmedi.", "model": "ignored",
    "questions": {
        "birim": {"type": "choice", "instructions": "Hangi birim?",
                  "criteria": {"kargo": "Teslimat", "fatura": None}},
        "acil": {"type": "noul", "instructions": "Acil mi?"},
        "puan": {"type": "score", "instructions": "Ne kadar kızgın?",
                 "criteria": ["sakin", "biraz", "çok"]},
    },
})  # fmt: skip


def body(content, cost=0.002):
    out = {"model": "served",
           "choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}],
           "usage": {"prompt_tokens": 100, "completion_tokens": 20}}  # fmt: skip
    return out


def adapter(handler, **kw):
    return StatedAdapter(
        model="m",
        base_url="https://api.test/v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kw,
    )


def test_one_request_per_item_with_a_strict_schema_and_typed_answers():
    seen = []
    stated = {"birim": {"kargo": 0.8, "fatura": 0.19}, "acil": {"true": 0.3, "false": 0.7},
              "puan": {"0": 0.1, "1": 0.6, "2": 0.3}}  # fmt: skip

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=body(stated))

    a = adapter(handler, extra={"reasoning_effort": "minimal"})
    result = a.answer(REQUEST)
    assert len(seen) == 1
    schema = seen[0]["response_format"]["json_schema"]["schema"]
    assert set(schema["required"]) == {"birim", "acil", "puan"}
    assert schema["properties"]["puan"]["required"] == ["0", "1", "2"]
    assert seen[0]["reasoning_effort"] == "minimal"
    # The endpoints take no temperature; none is sent unless asked for.
    assert "temperature" not in seen[0]
    answers = result.response.answers
    assert answers["birim"].choice == "kargo"
    assert sum(answers["birim"].probabilities.values()) == pytest.approx(1.0)
    assert result.diagnostics["birim"]["stated_sum"] == 0.99
    assert answers["acil"].noul == pytest.approx(0.3)
    assert answers["puan"].score == pytest.approx(1.2)
    assert result.confidence_source == {"birim": "max_probability", "puan": "max_probability"}
    assert a.info().prompt_version == PROMPT_VERSION


def test_a_missing_question_or_a_bad_distribution_is_an_error():
    a = adapter(lambda r: httpx.Response(200, json=body({"birim": {"kargo": 1, "fatura": 0}})))
    with pytest.raises(AdapterError, match="acil"):
        a.answer(REQUEST)
    bad = {"birim": {"kargo": 0, "fatura": 0}, "acil": {"true": 1, "false": 0},
           "puan": {"0": 1, "1": 0, "2": 0}}  # fmt: skip
    a = adapter(lambda r: httpx.Response(200, json=body(bad)))
    with pytest.raises(AdapterError, match="distribution"):
        a.answer(REQUEST)


def test_a_refusal_is_an_even_spread_marked_on_every_row():
    refused = {"model": "served", "choices": [{
        "finish_reason": "content_filter",
        "message": {"role": "assistant", "content": None, "refusal": None}}]}  # fmt: skip
    reason = "content_filter"
    a = adapter(lambda r: httpx.Response(200, json=refused))
    result = a.answer(REQUEST)
    answers = result.response.answers
    assert answers["acil"].noul == pytest.approx(0.5)
    assert answers["birim"].probabilities == pytest.approx({"kargo": 0.5, "fatura": 0.5})
    assert answers["puan"].score == pytest.approx(1.0)
    assert all(d["refused"] and d["finish_reason"] == reason for d in result.diagnostics.values())
