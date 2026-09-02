# kg-agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/KG-agent/blob/main/notebooks/quickstart.ipynb)

A knowledge graph used as an agent's **world model**, **planner state**, and **constraint on
hallucination** — rather than as a store to answer questions from.

Zero runtime dependencies (stdlib only). The model is a `Protocol`, not an SDK.

```bash
uv sync
uv run python -m kg_agent.demo        # walks through all three capabilities
uv run python -m kg_agent.demo_deps   # ...over a real uv.lock as the world model
uv run python -m kg_agent.demo --live   # ...using a real model via OpenRouter
uv run pytest -q
```

## 1. The graph is the agent's state

```
Experiment_42
 ├── uses_dataset → Dataset_A
 ├── tests        → Hypothesis_H3
 ├── produced     → Result_R7
 └── status       = failed
```

The agent runs `observe → update KG → query KG → plan → act → observe`, and keeps **no message
history**. Every model call is handed a freshly rendered slice of the graph:

```python
kg.context_for("Experiment_42", hops=2)
# world model (focus: Experiment_42; 2 hops; 6 entities, rev 14)
# Experiment_42 [Experiment] status=failed
#   --tests--> Hypothesis_H3
#   ...
```

So the context window carries a *view of the current world*, not the whole past. The graph is
what persists — `kg.save(path)` / `KnowledgeGraph.load(path)` round-trips nodes, edges, retracted
history and the change journal, so a long-horizon run survives across sessions.

Every edge carries provenance (`source`, `confidence`, `asserted_at`) and retraction is soft, so
you can always ask *why* the agent believes something.

## 2. The graph plans

`plan_for` backward-chains over the ontology's dependency predicates:

```python
plan = plan_for(kg, Goal("Hypothesis_A", "validate Hypothesis_A"))
print(plan.render())
# goal: validate Hypothesis_A
#   stage 0: secure_lab_access(Lab_D)
#   stage 1: acquire_instrument(Instrument_C)
#   stage 2: run_measurement(Measurement_B)
#   stage 3: evaluate_hypothesis(Hypothesis_A)
```

Already-satisfied subtrees are pruned, cycles raise `CyclicDependencyError`, independent work is
grouped into stages, and leaves with no known action are reported in `plan.blocked`. When a chain
dead-ends the agent asks the model for candidate sub-dependencies — but those proposals go through
verification before they touch the graph. **The model proposes; the graph constrains.**

## 3. The graph constrains hallucination

Before believing a generated statement, check it:

```python
verify(kg, Claim.parse("Protein_A inhibits Protein_B")).status
```

| status | meaning |
| --- | --- |
| `supported` | the edge is asserted in the graph |
| `entailed` | derivable via an inverse, a symmetric relation, or a transitive chain |
| `unknown` | no evidence either way |
| `contradicted` | the graph asserts something incompatible (`activates` vs `inhibits`) |
| `ill_formed` | unknown predicate/entity, or a domain–range violation |

A contradicted verdict carries the conflicting edges as `evidence`, so the agent can act on the
conflict instead of silently dropping the claim. `IngestPolicy` then decides what is written:
known claims are no-ops, unknown ones land provisionally at reduced confidence with
`source="llm"`, and contradicted or ill-formed ones never enter the world model.

This is only possible because relation *semantics* are declared — a bare triple store cannot know
that `inhibits` conflicts with `activates`:

```python
RelationSpec("inhibits", inverse="inhibited_by", domain="Protein", range="Protein",
             incompatible_with=frozenset({"activates"}))
RelationSpec("located_at", functional=True, range="Lab")   # rebinds instead of duplicating
RelationSpec("requires", transitive=True, dependency=True) # planner traverses this
```

`default_ontology()` ships a scientific vocabulary; build your own `Ontology` for another domain
(`lock_ontology()` in `kg_agent/demo_deps.py` is one, written from scratch in ~30 lines).

## A practical demo: upgrading a real dependency tree

Nothing in `kg_agent.demo_deps` is typed in by hand. The world model is parsed from a `uv.lock`
— this repo's own by default, or any uv project's — and the agent plans an upgrade over the real
tree, retries off the graph, refuses what the model gets wrong about the tree, and survives a
restart.

```bash
uv run python -m kg_agent.demo_deps
uv run python -m kg_agent.demo_deps --lock /path/to/other/uv.lock
```

| in the lockfile | in the graph |
| --- | --- |
| `[[package]]` | a `Package` node (or `Project` for the virtual root), `status=locked` |
| `version = "9.1.1"` | `pytest --pinned_to--> pytest==9.1.1` (`pinned_to` is functional) |
| `dependencies = [{ name = "packaging" }]` | `pytest --depends_on--> packaging` (transitive; inverse `required_by` materialised) |
| `marker = "sys_platform == 'win32'"` | a `marker` attribute on the dependency |
| `requires-python` + `.python-version` | a `Runtime` node the project `runs_on`, and an `Interpreter` that can satisfy it |

```
goal: upgrade kg-agent's dependency tree and resync
  stage 0: upgrade_package(colorama), upgrade_package(iniconfig), upgrade_package(packaging),
           upgrade_package(pluggy), upgrade_package(pygments), resolve(python) [blocked: no known action]
  stage 1: upgrade_package(pytest)
  stage 2: resync_project(kg-agent)
  blocked on: python
```

Stages follow dependency depth. No action satisfies a `Runtime`, so `python` is reported blocked
rather than guessed at; the agent asks the model, and of two proposals the graph writes
`python satisfied_by cpython-3.13` provisionally and refuses `python depends_on kg-agent` (a
`Project` is not a `Package`). A `pluggy` download that times out is retried by reading `status`
off the graph, not a counter in the process.

The upgrade step for `pytest` observes a fluent, wrong changelog summary. Each error is caught by a
different declared semantic:

| the claim | the rule | verdict |
| --- | --- | --- |
| `pytest depends_on tomli` | the policy refuses entities the lock never resolved | `ill_formed` — unknown entity |
| `packaging depends_on pytest` | `depends_on` is incompatible with its own inverse | `contradicted` — the graph holds `packaging required_by pytest` |
| `pytest pinned_to pytest==8.4.1` | `pinned_to` is functional | `ill_formed` under the strict policy; `contradicted` by the `9.1.1` pin even if new entities were allowed |

Section 4 stops the run after four steps with `pluggy` marked failed, saves the graph, loads it
into a brand-new agent with no transcript and no retry counters, and finishes the upgrade.

## Connecting a model (OpenRouter)

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run python -m kg_agent.openrouter     # one live call, end to end
uv run python -m kg_agent.demo --live    # the whole demo against a real model
```

`OpenRouterLLM` talks to the OpenAI-compatible `/api/v1/chat/completions` endpoint over
`urllib`, so there is still no dependency to install. The default model is
**`deepseek/deepseek-v4-flash-0731`**; override per call, or with `OPENROUTER_MODEL`.

```python
from kg_agent import KGAgent, KnowledgeGraph, OpenRouterLLM, default_ontology

kg = KnowledgeGraph(ontology=default_ontology())
llm = OpenRouterLLM(ontology=kg.ontology)         # ontology -> the allowed-predicate vocabulary
agent = KGAgent(kg, llm)

@agent.action("run_measurement")
def run_measurement(agent, step) -> str:
    return "measurement complete: Measurement_B produced Result_R12"   # fed back into observe()

run = agent.run(Goal("Hypothesis_A", "validate Hypothesis_A"))
print(run.render(), run.contradictions, llm.usage, sep="\n")
```

| setting | default | notes |
| --- | --- | --- |
| `model` | `deepseek/deepseek-v4-flash-0731` | or `OPENROUTER_MODEL` |
| `api_key` | `$OPENROUTER_API_KEY` | missing key raises at construction, not mid-run |
| `base_url` | `https://openrouter.ai/api/v1` | or `OPENROUTER_BASE_URL` |
| `ontology` | `None` | injects the predicate list (with domain/range) into the system prompt |
| `structured_output` | `True` | `response_format: json_schema`, auto-downgrading to plain text if the model rejects it |
| `max_retries` / `backoff` | `2` / `1.0s` | exponential, on 429/5xx/timeouts only — 4xx fails fast |
| `transport` | `None` | inject a callable to test without network (see `tests/test_openrouter.py`) |

Replies are parsed leniently — fenced JSON, a bare list, `{"s","p","o"}` keys, or one
`subject predicate object` per line — and anything unparseable yields no claims rather than an
exception. Usage is tallied on `llm.usage`.

The model has no write access. Everything it proposes returns as `Claim` objects and still has to
survive `verify` and the ingest policy, so a live run behaves exactly like the scripted one:

```
4. run_measurement(Measurement_B) -> ok
     observed: measurement complete: Measurement_B produced Result_R12, and Result_R12
               supports Hypothesis_A. Separately, Protein_C inhibits Protein_D.
     ingest: 0 known, 2 provisional, 1 rejected (1 contradicted)
  contradictions needing resolution:
    - [contradicted] Protein_C inhibits Protein_D -- graph asserts Protein_C --activates--> Protein_D
```

## Wiring a different provider

Implement three methods — no base class, no dependency:

```python
class MyLLM:
    def propose_claims(self, observation: str, context: str) -> list[Claim]: ...
    def propose_dependencies(self, node: str, context: str) -> list[Claim]: ...
    def choose_action(self, plan, context: str) -> str | None: ...
```

`ScriptedLLM` is the deterministic stand-in used by the demo and the tests.

## Layout

| file | role |
| --- | --- |
| `kg_agent/graph.py` | indexed triple store, provenance, journal, JSON persistence, `context_for` |
| `kg_agent/schema.py` | `RelationSpec` / `Ontology` — the semantics that make contradiction detectable |
| `kg_agent/verify.py` | `Claim`, `Status`, `verify`, `IngestPolicy` |
| `kg_agent/planner.py` | backward chaining, cycle detection, stages |
| `kg_agent/llm.py` | the model boundary (`LLM` Protocol, `ScriptedLLM`) |
| `kg_agent/openrouter.py` | `OpenRouterLLM` — live models over stdlib `urllib` |
| `kg_agent/agent.py` | the loop |
| `kg_agent/demo.py` | runnable walkthrough |
