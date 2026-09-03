import io
import json

import pytest
import urllib.error

from kg_agent import (DEFAULT_MODEL, Goal, KGAgent, KnowledgeGraph, OpenRouterError,
                      OpenRouterLLM, default_ontology, plan_for)
from kg_agent.openrouter import _parse_claims


def reply(content: str, usage: dict | None = None) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 4}}


class FakeTransport:
    """Stands in for the HTTP call; records payloads, replays canned replies."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


CLAIMS_JSON = json.dumps({"claims": [
    {"subject": "Experiment_42", "predicate": "tests", "object": "Hypothesis_H3",
     "confidence": 0.9},
    {"subject": "Protein_C", "predicate": "inhibits", "object": "Protein_D", "confidence": 0.4},
]})


def test_defaults_to_the_configured_model():
    llm = OpenRouterLLM(transport=FakeTransport(reply("{}")))
    assert llm.model == DEFAULT_MODEL == "deepseek/deepseek-v4-flash-0731"
    assert llm.base_url == "https://openrouter.ai/api/v1"


def test_model_and_key_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    llm = OpenRouterLLM()
    assert llm.model == "anthropic/claude-sonnet-5"
    assert llm.api_key == "sk-or-test"


def test_a_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError, match="no OpenRouter API key"):
        OpenRouterLLM()


def test_propose_claims_parses_the_reply_and_keeps_confidence():
    transport = FakeTransport(reply(CLAIMS_JSON))
    llm = OpenRouterLLM(ontology=default_ontology(), transport=transport)
    claims = llm.propose_claims("Experiment_42 tests Hypothesis_H3.", "# world model")
    assert [c.triple for c in claims] == [("Experiment_42", "tests", "Hypothesis_H3"),
                                          ("Protein_C", "inhibits", "Protein_D")]
    assert claims[0].confidence == 0.9 and claims[0].source == "llm"


def test_the_request_carries_the_ontology_and_the_graph_context():
    transport = FakeTransport(reply(CLAIMS_JSON))
    llm = OpenRouterLLM(ontology=default_ontology(), transport=transport)
    llm.propose_claims("something happened", "# world model (focus: Experiment_42)")
    payload = transport.payloads[0]
    assert payload["model"] == DEFAULT_MODEL
    assert payload["response_format"]["type"] == "json_schema"
    system, user = payload["messages"]
    assert "inhibits" in system["content"] and "object must be a Protein" in system["content"]
    assert "# world model (focus: Experiment_42)" in user["content"]
    assert "something happened" in user["content"]


def test_propose_dependencies_asks_only_for_dependency_predicates():
    transport = FakeTransport(reply(json.dumps({"claims": [
        {"subject": "Widget_X", "predicate": "satisfied_by", "object": "Spare_Part_9",
         "confidence": 0.7}]})))
    llm = OpenRouterLLM(ontology=default_ontology(), transport=transport)
    claims = llm.propose_dependencies("Widget_X", "# world model")
    assert [c.triple for c in claims] == [("Widget_X", "satisfied_by", "Spare_Part_9")]
    prompt = transport.payloads[0]["messages"][1]["content"]
    assert "requires" in prompt and "located_at" not in prompt


def test_choose_action_only_accepts_an_action_from_the_plan():
    kg = KnowledgeGraph(ontology=default_ontology())
    kg.add_node("Hypothesis_A", "Hypothesis")
    kg.add_node("Lab_D", "Lab")
    kg.add_node("Dataset_A", "Dataset")
    kg.assert_edge("Hypothesis_A", "requires", "Lab_D")
    kg.assert_edge("Hypothesis_A", "requires", "Dataset_A")
    plan = plan_for(kg, Goal("Hypothesis_A"))

    transport = FakeTransport(reply('{"action": "load_dataset"}'))
    good = OpenRouterLLM(transport=transport)
    assert good.choose_action(plan, "") == "load_dataset"
    prompt = transport.payloads[0]["messages"][1]["content"]
    assert "- load_dataset(Dataset_A)" in prompt and "- secure_lab_access(Lab_D)" in prompt
    assert "turn text into triples" not in transport.payloads[0]["messages"][0]["content"]

    exact = OpenRouterLLM(transport=FakeTransport(reply('{"action": "load_dataset(Dataset_A)"}')))
    assert exact.choose_action(plan, "") == "load_dataset(Dataset_A)"

    off_plan = OpenRouterLLM(transport=FakeTransport(reply('{"action": "launch_rocket"}')))
    assert off_plan.choose_action(plan, "") is None

    junk = OpenRouterLLM(transport=FakeTransport(reply("no idea")))
    assert junk.choose_action(plan, "") is None


def test_usage_accumulates_across_calls():
    transport = FakeTransport(reply(CLAIMS_JSON, {"prompt_tokens": 100, "completion_tokens": 20}))
    llm = OpenRouterLLM(transport=transport)
    llm.propose_claims("a", "")
    llm.propose_claims("b", "")
    assert llm.usage.requests == 2
    assert llm.usage.total_tokens == 240
    assert "240 tokens" in str(llm.usage)


def test_response_format_rejection_falls_back_to_plain_text():
    transport = FakeTransport(
        OpenRouterError("bad", status=400, body="response_format is not supported"),
        reply(CLAIMS_JSON),
    )

    def flaky(payload):
        transport.payloads.append(payload)
        response = transport.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    llm = OpenRouterLLM(transport=flaky)
    assert len(llm.propose_claims("x", "")) == 2
    assert not llm.structured_output
    assert "response_format" not in transport.payloads[1]


def test_other_400s_are_not_swallowed():
    def transport(payload):
        raise OpenRouterError("bad request", status=400, body="model not found")

    with pytest.raises(OpenRouterError, match="bad request"):
        OpenRouterLLM(transport=transport).propose_claims("x", "")


def test_an_empty_response_is_an_error():
    with pytest.raises(OpenRouterError, match="no choices"):
        OpenRouterLLM(transport=FakeTransport({"choices": []})).propose_claims("x", "")


class FakeHTTP:
    """Minimal urlopen stand-in so retry behaviour is testable offline."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else {"choices": []}
        if isinstance(outcome, Exception):
            raise outcome
        body = json.dumps(outcome).encode()

        class Handle:
            def read(self_inner):
                return body

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return Handle()


def http_error(code: int, body: str = "{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://openrouter.ai", code, "err", {},
                                  io.BytesIO(body.encode()))


def test_rate_limits_are_retried(monkeypatch):
    fake = FakeHTTP(http_error(429), reply(CLAIMS_JSON))
    monkeypatch.setattr("kg_agent.openrouter.urllib.request.urlopen", fake)
    llm = OpenRouterLLM(api_key="sk-or-test", backoff=0)
    assert len(llm.propose_claims("x", "")) == 2
    assert fake.calls == 2


def test_auth_failures_are_not_retried(monkeypatch):
    fake = FakeHTTP(http_error(401, '{"error": "no credit"}'))
    monkeypatch.setattr("kg_agent.openrouter.urllib.request.urlopen", fake)
    llm = OpenRouterLLM(api_key="sk-or-bad", backoff=0)
    with pytest.raises(OpenRouterError, match="HTTP 401"):
        llm.propose_claims("x", "")
    assert fake.calls == 1


def test_retries_are_exhausted_then_raised(monkeypatch):
    fake = FakeHTTP(http_error(503), http_error(503), http_error(503))
    monkeypatch.setattr("kg_agent.openrouter.urllib.request.urlopen", fake)
    llm = OpenRouterLLM(api_key="sk-or-test", backoff=0, max_retries=2)
    with pytest.raises(OpenRouterError, match="HTTP 503"):
        llm.propose_claims("x", "")
    assert fake.calls == 3


def test_headers_identify_the_app(monkeypatch):
    seen = {}

    def capture(request, timeout=None):
        seen.update(request.headers)
        seen["url"] = request.full_url
        return FakeHTTP(reply(CLAIMS_JSON))(request, timeout)

    monkeypatch.setattr("kg_agent.openrouter.urllib.request.urlopen", capture)
    OpenRouterLLM(api_key="sk-or-test", app_url="https://example.test").propose_claims("x", "")
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["Authorization"] == "Bearer sk-or-test"
    assert seen["X-openrouter-title"] == "kg-agent"
    assert seen["Http-referer"] == "https://example.test"


@pytest.mark.parametrize("raw, expected", [
    ('```json\n{"claims": [{"subject": "a", "predicate": "p", "object": "b",'
     ' "confidence": 1}]}\n```', [("a", "p", "b")]),
    ('[{"s": "a", "p": "p", "o": "b"}]', [("a", "p", "b")]),
    ('{"subject": "a", "predicate": "p", "object": "b"}', [("a", "p", "b")]),
    ("a p b\n- c q d", [("a", "p", "b"), ("c", "q", "d")]),
    ('{"claims": []}', []),
    ("", []),
    ("I could not find anything relevant in that text.", []),
    ('{"claims": [{"subject": "a", "predicate": "p"}]}', []),
])
def test_claim_parsing_is_tolerant(raw, expected):
    assert [c.triple for c in _parse_claims(raw)] == expected


def test_agent_runs_against_a_stubbed_openrouter():
    kg = KnowledgeGraph(ontology=default_ontology())
    kg.add_node("Hypothesis_A", "Hypothesis")
    kg.add_node("Measurement_B", "Measurement")
    kg.add_node("Protein_C", "Protein")
    kg.add_node("Protein_D", "Protein")
    kg.assert_edge("Hypothesis_A", "requires", "Measurement_B")
    kg.assert_edge("Protein_C", "activates", "Protein_D")

    llm = OpenRouterLLM(ontology=kg.ontology, transport=FakeTransport(reply(CLAIMS_JSON)))
    agent = KGAgent(kg, llm)
    agent.register("run_measurement", lambda a, step: "measurement complete")
    agent.register("evaluate_hypothesis", lambda a, step: "hypothesis evaluated")

    run = agent.run("Hypothesis_A")
    assert run.completed
    # the model's hallucination was refused; its grounded claim was written
    assert not kg.has_edge("Protein_C", "inhibits", "Protein_D")
    assert kg.has_edge("Experiment_42", "tests", "Hypothesis_H3")
    assert [v.claim.triple for v in run.contradictions][0] == ("Protein_C", "inhibits", "Protein_D")
