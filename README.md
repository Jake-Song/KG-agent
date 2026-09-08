# kg-agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/KG-agent/blob/main/notebooks/quickstart.ipynb)

**Ontology defines; agent executes.** A knowledge graph holds the agent's world model
and planner state. Declared semantics constrain proposed facts; the research workflow
also checks action contracts before recording successful execution.

Python 3.13+; zero runtime dependencies (stdlib only). The model boundary is a `Protocol`.

Use the [automated research cycle](#automated-research-with-codex-or-claude-code)
with a terminal-capable coding agent, or run the core graph demos:

```bash
uv sync
uv run python -m kg_agent.demo        # walks through all three capabilities
uv run python -m kg_agent.demo_replan # demonstrates replanning
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

The agent runs `observe → update KG → query KG → plan → choose → act → observe`, and keeps **no
message history**. Every model call is handed a freshly rendered slice of the graph:

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
#   stage 2: load_dataset(Dataset_B)
#   stage 3: run_measurement(Measurement_B)
#   stage 4: evaluate_hypothesis(Hypothesis_A)
```

Already-satisfied subtrees are pruned, cycles raise `CyclicDependencyError`, independent work is
grouped into stages, and leaves with no known action are reported in `plan.blocked`. When a chain
dead-ends the agent asks the model for candidate sub-dependencies — but those proposals go through
verification before they touch the graph. **The model proposes; the graph constrains.**

The demo follows one dependency chain, so `plan.ready` contains one actionable step
per turn. The agent acts, updates the graph, and replans:

```
goal: validate Hypothesis_A
  stage 0: secure_lab_access(Lab_D)
  stage 1: acquire_instrument(Instrument_C)
  ...
  turn 1: frontier of 1
  1. secure_lab_access(Lab_D) -> ok
  turn 2: frontier of 1
  2. acquire_instrument(Instrument_C) -> failed
```

A failed step is marked `status=failed` on the graph and falls into the next turn's frontier; a
turn spent unblocking a leaf is rendered as `turn N: blocked on X; asked the model, ...`.

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

The agent then **resolves** each contradiction against provenance. The default rule: if every
conflicting edge is itself a provisional model claim (`source="llm"`) with lower confidence than
the new one, the old edges are retracted (`reason="superseded by observation"`, inverse twins
included) and the claim is written; otherwise the graph's side is kept and the reason is reported.
`@agent.resolver` overrides that with your own `(agent, verdict) -> "keep" | "replace" | None`.
`run.resolutions` lists every decision and `run.unresolved` the contradictions that were kept:

```
5. run_measurement(Measurement_B) -> ok
     ingest: 0 known, 2 provisional, 2 rejected (2 contradicted)
     resolved: replaced: Protein_G inhibits Protein_H -- every conflicting edge is provisional ('llm') ...
contradictions needing resolution:
  - [contradicted] Protein_C inhibits Protein_D -- ... (kept: Protein_C --activates--> Protein_D comes from 'observation', ...)
```

This is only possible because relation *semantics* are declared — a bare triple store cannot know
that `inhibits` conflicts with `activates`:

```python
RelationSpec("inhibits", inverse="inhibited_by", domain="Protein", range="Protein",
             incompatible_with=frozenset({"activates"}))
RelationSpec("located_at", functional=True, range="Lab")   # rebinds instead of duplicating
RelationSpec("requires", transitive=True, dependency=True) # planner traverses this
```

`default_ontology()` ships a scientific vocabulary. Build your own `Ontology` for
another domain; [ResearchOntology](kg_agent/research_domain.py) adds the typed
relations and executable contracts used by the research cycle below.

## Automated research with Codex or Claude Code

Run actual CPU experiments through a persistent CLI. The coding agent reads current
evidence and proposes a bounded configuration; the project validates it, executes
the experiment, and records measured results. The CLI does not launch a coding agent
or require its own model-provider API key.

```text
Inspect state → propose hypothesis → validate contracts → fit model
      ↑                                                   ↓
      └──────── choose next experiment ← evaluate validation MSE
                                             ↓
                        finalize validation winner → evaluate test once
```

| Component | Responsibility |
| --- | --- |
| Ontology and contracts | Allowed configurations, typed relations, prerequisites, required evidence |
| Knowledge graph | Hypotheses, experiments, models, evaluations, dependencies, and provenance |
| Planner | Order fitting before evaluation using graph dependencies |
| Coding agent | Explain a hypothesis, submit its configuration, invoke permitted commands |
| Fixed evaluator | Fit coefficients, compute metrics, and determine the measured conclusion |

### Start or resume a session

```bash
uv run python -m kg_agent.research init --dir research-runs/my-study --budget 5
uv run python -m kg_agent.research run --dir research-runs/my-study
uv run python -m kg_agent.research status --dir research-runs/my-study
```

The first `run` executes the linear baseline. For an existing directory, start with
`status` instead of `init`. Ask your coding agent:

> Read docs/research-cycle.md and conduct a research session in
> research-runs/my-study with a budget of five trials. Use the project CLI for all
> proposals, experiments, and evaluations. Start with the baseline; choose each
> subsequent hypothesis deliberately from accumulated validation evidence. Continue
> until the budget is used or no useful permitted hypothesis remains, then finalize
> and summarize the evidence and limitations. If the session exists, inspect its
> status and resume it.

[AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) point to the shared
[research instructions](docs/research-cycle.md).

### Submit an experiment

For example, after measuring the linear baseline, test whether adding curvature
reduces validation error:

```bash
cat > research-runs/my-study/proposal.json <<'JSON'
{
  "id": "quadratic",
  "hypothesis": "A quadratic term will lower validation MSE by at least 1%",
  "rationale": "Test whether curvature explains error left by the linear baseline",
  "degree": 2,
  "penalty": 0
}
JSON

uv run python -m kg_agent.research propose --dir research-runs/my-study --file research-runs/my-study/proposal.json
uv run python -m kg_agent.research run --dir research-runs/my-study
uv run python -m kg_agent.research status --dir research-runs/my-study
```

Choose subsequent hypotheses from the returned evidence. The benchmark permits
polynomial degrees 1–5 and ridge penalties `0`, `0.001`, `0.01`, `0.1`, or `1`.
The five-trial default budget includes the baseline. Data splits and seed are fixed
at initialization; only validation MSE selects the winner.

| Command | What it does |
| --- | --- |
| `init` | Freeze settings, create the graph, and queue the baseline |
| `status` | Show all trials, contracts, remaining budget, and permitted next commands |
| `propose` | Validate a unique configuration and record its hypothesis and comparison target |
| `run` | Fit and evaluate the pending trial, validate evidence, and save its conclusion |
| `finalize` | Select the validation winner, evaluate its saved model on test data, and freeze the study |

Every command requires `--dir`. Operational output is JSON with `ok`, `description`,
and either `result` or `error`. Exit codes are 0 for success, 2 for invalid requests,
and 3 for execution/storage/lock failures. Commands never prompt interactively.

### Inspect every step

Each trial returned by `status`, `propose`, or `run` includes ordered explanations:

1. **Hypothesis:** what the agent predicts and why.
2. **Contract:** allowed configuration and study rules.
3. **Plan:** why fitting must precede evaluation.
4. **Fit:** training setup, evidence requirements, and saved coefficients.
5. **Evaluate:** validation procedure and measured MSE.
6. **Compare:** reference trial, relative improvement, and whether the 1% criterion was met.
7. **Persist:** saved status and budget usage.

Descriptions distinguish pending, completed, failed, and skipped actions. They are
derived from saved evidence, not timestamped execution logs. To read the JSON more easily:

```bash
uv run python -m kg_agent.research status --dir research-runs/my-study | uv run python -m json.tool
```

### Finalize and inspect the report

When the budget is exhausted, or no useful permitted hypothesis remains:

```bash
uv run python -m kg_agent.research finalize --dir research-runs/my-study
```

The session directory contains `state.json` (authoritative graph and experiment
records) and `report.md` (readable results with descriptions of every step).
Local sessions under `research-runs/` are gitignored.

Interrupted trials resume under the same ID without consuming another slot.
Completed trials cannot be rerun. Repeating `finalize` regenerates the report from
saved scores without reevaluating the test split. To rerun a finalized study, choose
a new session directory. CLI locking requires a POSIX system such as Linux or macOS.

### Example: five-trial run

A seed-42 run and a fresh rerun produced these validation results:

| Trial | Degree | Ridge penalty | Validation MSE | Improvement over prior best |
| --- | ---: | ---: | ---: | ---: |
| Linear baseline | 1 | 0 | 0.35553404 | — |
| Quadratic | 2 | 0 | 0.00975547 | 97.256% |
| Quadratic + mild ridge | 2 | 0.001 | 0.00974331 | 0.125% |
| Cubic | 3 | 0 | 0.00973230 | 0.113% |
| Cubic + mild ridge | 3 | 0.001 | **0.00971334** | 0.195% |

The winner's final test MSE was **0.01014440**. Adding the quadratic term met the
1% improvement criterion; the later gains did not. The lowest-MSE trial still wins
even when its margin falls below that criterion.

These results demonstrate the mechanics and reproducibility of the cycle on a
small synthetic benchmark: 80 training, 40 validation, and 40 test observations.
They do not establish statistical significance or real-world generalization. The
same-seed rerun is not an independent replication. The CLI enforces its own action
boundary; it is not a sandbox against an agent that edits the source or saved state.

See [the full guide](docs/research-cycle.md) for persistence, error handling, and
research-session rules. The existing `KGAgent` / OpenRouter integration below is a
separate way to operate the core graph loop.

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
from kg_agent import Goal, KGAgent, KnowledgeGraph, OpenRouterLLM, default_ontology

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
survive `verify` and the ingest policy, and its `choose_action` reply (the prompt lists the ready
steps as `action(node)` entries) is only honoured when it names one of them — so a live run behaves
exactly like the scripted one:

```
5. run_measurement(Measurement_B) -> ok
     observed: measurement complete: Measurement_B produced Result_R12, and Result_R12
               supports Hypothesis_A. Separately, Protein_C inhibits Protein_D, ...
     ingest: 0 known, 2 provisional, 2 rejected (2 contradicted)
     resolved: replaced: Protein_G inhibits Protein_H -- every conflicting edge is provisional ('llm') ...
  contradictions needing resolution:
    - [contradicted] Protein_C inhibits Protein_D -- graph asserts Protein_C --activates--> Protein_D (kept: ...)
```

## Wiring a different provider

Implement three methods — no base class, no dependency:

```python
class MyLLM:
    def propose_claims(self, observation: str, context: str) -> list[Claim]: ...
    def propose_dependencies(self, node: str, context: str) -> list[Claim]: ...
    def choose_action(self, plan, context: str) -> str | None: ...   # a bare action or "action(node)" from plan.ready
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
| `kg_agent/agent.py` | the loop: model choice, stage execution, contradiction resolution |
| `kg_agent/demo.py` | runnable walkthrough |
| `kg_agent/demo_replan.py` | replanning walkthrough |
| `kg_agent/research.py` | persistent CLI, research execution, and final report |
| `kg_agent/research_domain.py` | research ontology and executable contracts |
| `kg_agent/research_experiment.py` | fixed synthetic data, polynomial fitting, and MSE evaluation |
| `kg_agent/research_explain.py` | evidence-derived step descriptions |
| `docs/research-cycle.md` | instructions for conducting research with a coding agent |
