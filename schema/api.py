"""Request and response bodies of the evaluation endpoint.

Same field names as the public typed-decision contract (docs.typesafe.ai,
page api, read 2026-09-19): a request is a state, a model name and a map of
named questions; a response carries one answer per question under the same
names, plus token usage.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schema.questions import Answer, JsonContent, Question


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: JsonContent
    model: str = Field(min_length=1)
    # The caller picks each key. Keys are not shown to the model.
    questions: dict[str, Question]

    @model_validator(mode="after")
    def _check_questions(self) -> Request:
        if not self.questions:
            raise ValueError("a request needs at least one question")
        if any(not key.strip() for key in self.questions):
            raise ValueError("question ids must not be empty")
        return self


class Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Response(BaseModel):
    # Unknown top-level fields from a remote server are ignored, not fatal.
    model_config = ConfigDict(extra="ignore")

    model: str
    answers: dict[str, Answer]
    usage: Usage | None = None


def check_response(request: Request, response: Response) -> None:
    """Raise ValueError unless the response answers exactly the questions asked.

    Checks the ids, the answer types, and that each distribution covers
    exactly the options or levels the question defined.
    """
    asked, answered = set(request.questions), set(response.answers)
    if asked != answered:
        missing, extra = sorted(asked - answered), sorted(answered - asked)
        raise ValueError(f"answers do not match questions: missing {missing}, extra {extra}")
    for qid, question in request.questions.items():
        answer = response.answers[qid]
        if answer.type != question.type:
            raise ValueError(f"{qid}: asked a {question.type} question, got a {answer.type} answer")
        if question.type == "choice" and set(answer.probabilities) != set(question.criteria):
            raise ValueError(f"{qid}: probabilities do not cover the options exactly")
        if question.type == "score":
            if len(answer.legend) != len(question.criteria):
                raise ValueError(f"{qid}: legend does not match the number of levels")
            for k, description in enumerate(question.criteria):
                if answer.legend[str(k)] != description:
                    raise ValueError(f"{qid}: legend entry {k} differs from the question")
