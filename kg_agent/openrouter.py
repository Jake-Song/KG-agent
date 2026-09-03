"""An :class:`~kg_agent.llm.LLM` backed by OpenRouter.

Stdlib only -- ``urllib.request`` against the OpenAI-compatible
``/api/v1/chat/completions`` endpoint, so the package keeps zero runtime
dependencies.

    export OPENROUTER_API_KEY=sk-or-...
    uv run python -m kg_agent.openrouter        # live connection check

The model never writes to the graph directly: whatever it proposes comes back
as :class:`~kg_agent.verify.Claim` objects and still has to survive
:func:`~kg_agent.verify.verify` and the ingest policy.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .verify import Claim

if TYPE_CHECKING:  # pragma: no cover
    from .planner import Plan
    from .schema import Ontology

__all__ = ["DEFAULT_MODEL", "DEFAULT_BASE_URL", "OpenRouterError", "Usage", "OpenRouterLLM"]

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

CLAIMS_SCHEMA: dict[str, Any] = {
    "name": "claims",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["subject", "predicate", "object", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """\
You maintain a knowledge graph that is an autonomous agent's world model.

Your only job is to turn text into triples. Rules:
- Reply with JSON: {"claims": [{"subject", "predicate", "object", "confidence"}]}.
- Reuse the exact entity ids that appear in the world-model context whenever you mean the
  same entity. Invent a new id only for something genuinely new.
- Use only the predicates listed below. If nothing fits, emit no claim for that statement.
- confidence is 0.0-1.0: how sure you are the text asserts this, not how plausible it sounds.
- Emit only what the text actually states. Do not infer, elaborate, or add background knowledge.
- Every claim is checked against the graph before it is accepted, and anything that
  contradicts existing edges is discarded. Guessing costs you; omitting is free.

Return an empty list when the text asserts no relationship."""

CHOOSE_SYSTEM_PROMPT = """\
You pick which ready action an autonomous agent should take next.

Every listed action is admissible and its prerequisites are met; you only choose the order.
Reply with JSON: {"action": "<one entry from the list, exactly as written>"}."""

ACTION_SCHEMA: dict[str, Any] = {
    "name": "action",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
        "additionalProperties": False,
    },
}


class OpenRouterError(RuntimeError):
    """A non-recoverable OpenRouter response."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(slots=True)
class Usage:
    """Token accounting across the life of one client."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, raw: dict[str, Any] | None) -> None:
        self.requests += 1
        if raw:
            self.prompt_tokens += int(raw.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(raw.get("completion_tokens", 0) or 0)

    def __str__(self) -> str:
        return (f"{self.requests} requests, {self.prompt_tokens} prompt + "
                f"{self.completion_tokens} completion = {self.total_tokens} tokens")


class OpenRouterLLM:
    """Satisfies the :class:`~kg_agent.llm.LLM` protocol using OpenRouter."""

    def __init__(self, model: str | None = None, *,
                 api_key: str | None = None,
                 base_url: str | None = None,
                 ontology: Ontology | None = None,
                 temperature: float = 0.0,
                 max_tokens: int = 1024,
                 timeout: float = 60.0,
                 max_retries: int = 2,
                 backoff: float = 1.0,
                 app_url: str | None = None,
                 app_title: str = "kg-agent",
                 structured_output: bool = True,
                 extra_body: dict[str, Any] | None = None,
                 transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> None:
        self.model = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.ontology = ontology
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.app_url = app_url or os.environ.get("OPENROUTER_APP_URL")
        self.app_title = app_title
        self.structured_output = structured_output
        self.extra_body = dict(extra_body or {})
        self._transport = transport
        self.usage = Usage()
        self.last_response: dict[str, Any] | None = None
        if self._transport is None and not self.api_key:
            raise OpenRouterError(
                "no OpenRouter API key: pass api_key=... or set OPENROUTER_API_KEY")

    # ------------------------------------------------------------- LLM protocol

    def propose_claims(self, observation: str, context: str) -> list[Claim]:
        prompt = (f"{self._context_block(context)}"
                  f"New observation:\n{observation}\n\n"
                  "Extract the relationships this observation asserts.")
        return self._claims(prompt)

    def propose_dependencies(self, node: str, context: str) -> list[Claim]:
        predicates = sorted(self.ontology.dependency_predicates) if self.ontology else ["requires"]
        prompt = (f"{self._context_block(context)}"
                  f"Planning is stuck: '{node}' has to be satisfied but the graph records no way "
                  f"to satisfy it.\n\nPropose the prerequisites of '{node}', using only "
                  f"{', '.join(predicates)} or satisfied_by. Propose nothing if you cannot "
                  "name a concrete prerequisite.")
        return self._claims(prompt)

    def choose_action(self, plan: Plan, context: str) -> str | None:
        """Let the model pick which ready step goes first; anything off-plan is ignored.

        Options are listed as ``action(node)`` so steps sharing an action stay
        distinguishable; a bare action name is accepted too.  The reply is
        returned as given once :meth:`Plan.find` resolves it to a ready step.
        """
        options = plan.ready
        if len(options) < 2:
            return None
        listing = "\n".join(f"- {step}" for step in options)
        prompt = (f"{self._context_block(context)}"
                  f"Goal: {plan.goal}\n\nReady actions (any order is valid):\n{listing}\n\n"
                  'Reply with JSON: {"action": "<one entry from the list, exactly as written>"}.')
        raw = self._chat(prompt, schema=ACTION_SCHEMA, system=CHOOSE_SYSTEM_PROMPT)
        try:
            choice = json.loads(_strip_fences(raw)).get("action")
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(choice, str):
            return None
        return choice.strip() if plan.find(choice) is not None else None

    # ------------------------------------------------------------------ helpers

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Low-level single-turn completion, for callers building their own prompts."""
        return self._chat(prompt, schema=None, system=system)

    def _context_block(self, context: str) -> str:
        return f"Current world model:\n{context}\n\n" if context else ""

    def _system_prompt(self) -> str:
        return f"{SYSTEM_PROMPT}\n\n{self._vocabulary()}"

    def _vocabulary(self) -> str:
        if self.ontology is None or not self.ontology.relations:
            return "Allowed predicates: any snake_case verb phrase."
        lines = ["Allowed predicates:"]
        for name, spec in sorted(self.ontology.relations.items()):
            bits = []
            if spec.domain:
                bits.append(f"subject must be a {spec.domain}")
            if spec.range:
                bits.append(f"object must be a {spec.range}")
            if spec.functional:
                bits.append("at most one object")
            if spec.symmetric:
                bits.append("symmetric")
            note = "; ".join(filter(None, [spec.description, *bits]))
            lines.append(f"- {name}" + (f" -- {note}" if note else ""))
        return "\n".join(lines)

    def _claims(self, prompt: str) -> list[Claim]:
        return _parse_claims(self._chat(prompt, schema=CLAIMS_SCHEMA))

    def _chat(self, prompt: str, *, schema: dict[str, Any] | None,
              system: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if self.structured_output:
            payload["response_format"] = ({"type": "json_schema", "json_schema": schema}
                                          if schema else {"type": "json_object"})
        try:
            response = self._send(payload)
        except OpenRouterError as exc:
            # Some models reject response_format; fall back to plain text once.
            if not (self.structured_output and exc.status == 400
                    and ("response_format" in exc.body or "json_schema" in exc.body)):
                raise
            self.structured_output = False
            payload.pop("response_format", None)
            response = self._send(payload)

        self.last_response = response
        self.usage.add(response.get("usage"))
        choices = response.get("choices") or []
        if not choices:
            raise OpenRouterError(f"no choices in response: {json.dumps(response)[:400]}")
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(payload)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.app_title,
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        request = urllib.request.Request(f"{self.base_url}/chat/completions",
                                         data=body, headers=headers, method="POST")
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                    parsed = json.loads(handle.read().decode("utf-8"))
                if "error" in parsed and not parsed.get("choices"):
                    detail = parsed["error"]
                    raise OpenRouterError(f"OpenRouter error: {detail}",
                                          status=int(detail.get("code") or 0) or None,
                                          body=json.dumps(detail))
                return parsed
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                error = OpenRouterError(f"HTTP {exc.code} from OpenRouter: {detail[:400]}",
                                        status=exc.code, body=detail)
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise error from exc
                last = error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = OpenRouterError(f"OpenRouter request failed: {exc}")
            if attempt < self.max_retries:
                time.sleep(self.backoff * (2 ** attempt))
        raise last if last else OpenRouterError("OpenRouter request failed")

    def __repr__(self) -> str:
        return f"OpenRouterLLM(model={self.model!r}, {self.usage})"


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_claims(text: str) -> list[Claim]:
    """Tolerant parsing: JSON first, then one-triple-per-line."""
    cleaned = _strip_fences(text)
    if not cleaned:
        return []
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return _parse_lines(cleaned)

    items: Sequence[Any]
    if isinstance(payload, dict):
        for key in ("claims", "triples", "edges", "facts"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        else:
            items = [payload] if {"subject", "predicate", "object"} <= set(payload) else []
    elif isinstance(payload, list):
        items = payload
    else:
        return []

    claims: list[Claim] = []
    for item in items:
        if isinstance(item, str):
            claims.extend(_parse_lines(item))
            continue
        if not isinstance(item, dict):
            continue
        subject = item.get("subject") or item.get("s")
        predicate = item.get("predicate") or item.get("p") or item.get("relation")
        obj = item.get("object") or item.get("o")
        if not (subject and predicate and obj):
            continue
        confidence = item.get("confidence")
        claims.append(Claim(str(subject).strip(), str(predicate).strip(), str(obj).strip(),
                            text=item.get("text"), source="llm",
                            confidence=float(confidence) if isinstance(confidence, (int, float))
                            else None))
    return claims


def _parse_lines(text: str) -> list[Claim]:
    claims: list[Claim] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if not line:
            continue
        try:
            claims.append(Claim.parse(line))
        except ValueError:
            continue
    return claims


def main() -> None:  # pragma: no cover - live connection check
    """`python -m kg_agent.openrouter` -- one real call, end to end."""
    from .graph import KnowledgeGraph
    from .schema import default_ontology
    from .verify import IngestPolicy, ingest

    kg = KnowledgeGraph(ontology=default_ontology())
    kg.add_node("Experiment_42", "Experiment", status="running")
    kg.add_node("Dataset_A", "Dataset")
    kg.add_node("Hypothesis_H3", "Hypothesis")
    kg.add_node("Protein_C", "Protein")
    kg.add_node("Protein_D", "Protein")
    kg.assert_edge("Experiment_42", "uses_dataset", "Dataset_A")
    kg.assert_edge("Protein_C", "activates", "Protein_D")

    llm = OpenRouterLLM(ontology=kg.ontology)
    print(f"model: {llm.model}\nendpoint: {llm.base_url}/chat/completions\n")

    observation = ("Experiment_42 finished and tests Hypothesis_H3; it produced Result_R7. "
                   "Protein_C inhibits Protein_D.")
    print(f"observation: {observation}\n")
    proposals = llm.propose_claims(observation, kg.context_for("Experiment_42"))
    if not proposals:
        print("the model proposed nothing -- check the model id and key")
        return
    report = ingest(kg, proposals, policy=IngestPolicy(unknown_confidence=0.6))
    for verdict in report.verdicts:
        print(f"  [{verdict.status:<13}] {verdict.claim}")
        if verdict.evidence:
            print(f"                  evidence: {verdict.evidence[0]}")
    print(f"\n{report.summary()}")
    print(f"usage: {llm.usage}")
    print(f"graph unchanged where it mattered: Protein_C activates Protein_D is "
          f"{kg.has_edge('Protein_C', 'activates', 'Protein_D')}, "
          f"inhibits is {kg.has_edge('Protein_C', 'inhibits', 'Protein_D')}")


if __name__ == "__main__":  # pragma: no cover
    main()
