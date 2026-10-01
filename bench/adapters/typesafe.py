"""Adapter for the hosted typed-decision API.

One POST answers a whole request: the endpoint takes the state and the
whole question map and returns one typed answer per question, with its own
confidence. Follows docs.typesafe.ai, pages api, sdk/python/api/constants
and sdk/python/api/retries, read 2026-09-19.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from bench.adapters.base import AdapterError, AdapterResult, ModelInfo
from schema.api import Request, Response, check_response

# Page sdk/python/api/constants.
API_KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 10.0
# Page api, "Evaluation endpoint".
ENDPOINT_PATH = "/v1/systemone"
# Page api, "Handling rate limits": these two statuses are worth another try.
RETRY_STATUSES = frozenset({429, 529})
# Page sdk/python/api/retries: the SDK's default backoff, doubled each time,
# with up to backoff_jitter of each delay subtracted at random.
BACKOFF_INITIAL = 0.5
BACKOFF_MAX = 5.0
BACKOFF_JITTER = 0.25
# The API returns its own confidence on choice and score answers.
CONFIDENCE_SOURCE = "model"
# How much of a server message an error carries.
MAX_SERVER_MESSAGE = 500


class TypeSafeAdapter:
    """Answers one Request with one call to the decision endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = DEFAULT_TIMEOUT,
        path_used: str = "typesafe api",
    ) -> None:
        key = api_key or os.environ.get(API_KEY_ENV) or ""
        if not key:
            raise AdapterError(f"no API key: pass api_key or set {API_KEY_ENV}")
        self.path_used = path_used
        self._api_key = key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(timeout=timeout) if client is None else client

    def info(self) -> ModelInfo:
        return ModelInfo(
            adapter="typesafe",
            model=self.model,
            revision=self.model,
            path_used=self.path_used,
        )

    def answer(self, request: Request) -> AdapterResult:
        http, latency_ms = self._post(_body(request, self.model))
        response = self._parse(http, request)
        sums = _returned_sums(http)
        named = _returned_choices(http)
        result = AdapterResult(
            response=response,
            latency_ms=latency_ms,
            per_question_ms=None,
            confidence_source={
                qid: CONFIDENCE_SOURCE
                for qid, answer in response.answers.items()
                if answer.type != "noul"
            },
            diagnostics={
                qid: {
                    "served_model": response.model,
                    **({"probability_sum_returned": sums[qid]} if qid in sums else {}),
                    **(
                        {"choice_returned": named[qid]}
                        if named.get(qid, getattr(response.answers[qid], "choice", None))
                        != getattr(response.answers[qid], "choice", None)
                        else {}
                    ),
                }
                for qid in response.answers
            },  # fmt: skip
        )
        return result

    def _post(self, body: dict[str, Any]) -> tuple[httpx.Response, float]:
        """Send the body, retrying rate limits and transport failures.

        Returns the successful response and how long its call took, in
        milliseconds.
        """
        url = f"{self.base_url}{ENDPOINT_PATH}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        delay = BACKOFF_INITIAL
        attempt = 0
        while True:
            last = attempt >= self.max_retries
            started = time.perf_counter()
            try:
                http = self._client.post(url, json=body, headers=headers)
            except httpx.HTTPError as error:
                if last:
                    raise AdapterError(f"the request failed: {self._safe(str(error))}") from error
                wait = self._jittered(delay)
            else:
                latency_ms = (time.perf_counter() - started) * 1000.0
                if http.is_success:
                    return http, latency_ms
                if http.status_code not in RETRY_STATUSES or last:
                    raise AdapterError(
                        f"the API returned {http.status_code}: {self._safe(_server_message(http))}"
                    )
                after = _retry_after(http)
                wait = self._jittered(delay) if after is None else after
            self._sleep(wait)
            delay = min(delay * 2, BACKOFF_MAX)
            attempt += 1

    def _parse(self, http: httpx.Response, request: Request) -> Response:
        """Read the body as a Response and check it answers what was asked."""
        try:
            response = Response.model_validate(_rescaled(http.json()))
        except ValueError as error:
            # Raised from None: a validation error prints the input it rejected,
            # and a chained cause would show in a traceback unredacted.
            raise AdapterError(
                f"the API body is not a response: {self._safe(str(error))}"
            ) from None
        try:
            check_response(request, response)
        except ValueError as error:
            raise AdapterError(self._safe(str(error))) from None
        return response

    def _jittered(self, delay: float) -> float:
        return delay * (1.0 - random.random() * BACKOFF_JITTER)

    def _safe(self, text: str) -> str:
        """Text for an error message, with the API key taken out of it."""
        return text.replace(self._api_key, "[redacted]")


def _body(request: Request, model: str) -> dict[str, Any]:
    """The request as JSON for the endpoint, under the adapter's model.

    Optional fields the caller left unset are left out rather than sent as
    null. A choice option with no description is a null inside `criteria`
    and stays, because the contract asks for one key per option.
    """
    body = request.model_dump(mode="json")
    body["model"] = model
    for question in body["questions"].values():
        for field in [name for name, value in question.items() if value is None]:
            del question[field]
        if question["type"] == "noul" and "criteria" in question:
            criteria = question["criteria"]
            for name in [name for name, value in criteria.items() if value is None]:
                del criteria[name]
    return body


# The API rounds probabilities to two decimals, so a distribution can sum to 0.99 or 1.01;
# up to ten options at half a hundredth each can drift by 0.05 at most.
ROUNDING_SLACK = 0.05


def _rescaled(body: Any) -> Any:
    """The body with each rounded distribution rescaled to sum to 1; order is unchanged.

    The answer's own score and confidence are kept as the API returned them. A sum
    further from 1 than rounding explains is left for validation to refuse.
    """
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        return body
    answers = {}
    for qid, answer in body["answers"].items():
        probs = answer.get("probabilities") if isinstance(answer, dict) else None
        if isinstance(probs, dict) and probs:
            total = sum(float(v) for v in probs.values())
            if total > 0 and abs(total - 1.0) <= ROUNDING_SLACK:
                answer = {
                    **answer,
                    "probabilities": {k: float(v) / total for k, v in probs.items()},
                }
            # Rounding can leave the named choice a hundredth below another option; the
            # scores read the probabilities, so the choice follows them within that hundredth.
            choice = answer.get("choice")
            if choice in probs:
                best = max(float(v) for v in probs.values())
                if float(probs[choice]) < best and best - float(probs[choice]) <= 0.011:
                    now = answer["probabilities"]
                    answer = {**answer, "choice": max(now, key=now.get)}
        answers[qid] = answer
    return {**body, "answers": answers}


def _returned_sums(http: httpx.Response) -> dict[str, float]:
    """Per question, the sum of the probabilities as returned, when it was not exactly 1."""
    try:
        body = http.json()
    except ValueError:
        return {}
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        return {}
    out = {}
    for qid, answer in body["answers"].items():
        probs = answer.get("probabilities") if isinstance(answer, dict) else None
        if isinstance(probs, dict) and probs:
            total = sum(float(v) for v in probs.values())
            if total != 1.0:
                out[qid] = round(total, 6)
    return out


def _returned_choices(http: httpx.Response) -> dict[str, str]:
    """Per choice question, the option the API named, kept when rounding moved it."""
    try:
        body = http.json()
    except ValueError:
        return {}
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        return {}
    return {
        qid: a["choice"]
        for qid, a in body["answers"].items()
        if isinstance(a, dict) and isinstance(a.get("choice"), str)
    }


def _server_message(http: httpx.Response) -> str:
    """What the server said about an error, short enough to put in a message."""
    try:
        body = http.json()
    except ValueError:
        return http.text.strip()[:MAX_SERVER_MESSAGE]
    if isinstance(body, dict):
        for key in ("detail", "error", "message"):
            said = body.get(key)
            if isinstance(said, str):
                return said[:MAX_SERVER_MESSAGE]
    return json.dumps(body, ensure_ascii=False)[:MAX_SERVER_MESSAGE]


def _retry_after(http: httpx.Response) -> float | None:
    """The delay the server asked for, in seconds, or None if it asked for none.

    Page sdk/python/api/retries: the SDK honours `retry-after-ms` and
    `Retry-After` by default. Only the seconds form of `Retry-After` is
    read here; an HTTP-date falls back to the backoff.
    """
    for header, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        raw = http.headers.get(header)
        if raw is None:
            continue
        try:
            seconds = float(raw) / divisor
        except ValueError:
            continue
        return min(max(seconds, 0.0), BACKOFF_MAX)
    return None
