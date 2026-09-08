# Automated research with a coding agent

This project supplies a fixed experiment environment and persistent evidence. Codex,
Claude Code, or another terminal-capable agent drives it through a JSON CLI. No
provider SDK, API key, agent subprocess launcher, or GPU is needed by the project.
The coding agent uses its own existing authentication and terminal permissions.

The cycle is:

```text
inspect state → propose a hypothesis → validate → fit → evaluate → inspect state
                                                     ↓
                                  finalize the validation winner → test once
```

## Ask your agent

From the repository, give either coding agent this task:

> Read docs/research-cycle.md and conduct a research session in
> research-runs/demo with a budget of five trials. Use the project CLI for all
> proposals, experiments, and evaluations. Start with the baseline; choose each
> subsequent hypothesis deliberately from the accumulated validation evidence.
> Continue until the budget is used or no useful permitted hypothesis remains,
> then finalize and summarize the evidence and limitations. If the session already
> exists, inspect its status and resume it.

The session survives coding-agent context changes and process restarts. Run
`status` to recover current facts and permitted commands. A running coding agent
is needed to choose subsequent hypotheses: the CLI does not launch one itself.

## Research instructions

These instructions apply when conducting a research session, not when developing
or testing the research implementation.

1. Run `status` for an existing session; otherwise `init` it. Inspect the returned
   contracts, evidence, budget, and `next_commands` before acting.
2. Run the pending baseline. Explain why the next configuration is scientifically
   relevant to the observed validation result. Change one factor at a time when
   useful; do not randomly select methods or claim unseen evidence.
3. Write a proposal JSON file, submit it with `propose`, then execute `run`.
   Use unique descriptive IDs. The hypothesis predicts at least 1% lower validation
   MSE than the best trial at submission time.
4. Retain failed and non-improving trials. Use their evidence to choose the next
   proposal. An unsuccessful hypothesis is still a completed experiment.
5. Stop at the budget, when no useful permitted proposal remains, or when the
   baseline fails. Finalize only when permitted. Report a failed baseline as blocked.
6. Never edit session state, the benchmark, the ontology, or the evaluator during
   a study. Do not calculate or inspect test results before finalization. Submit
   configurations and rationale only; the evaluator supplies measured evidence.
7. Report validation comparisons as observations on this synthetic benchmark, not
   statistical proof or evidence of general real-world model improvement.

The CLI enforces its command boundary, not a security sandbox against an agent
that can modify local source files or state. Use a separate study for a changed
benchmark, seed, budget, or hypothesis space.

## Runnable walkthrough

```bash
uv run python -m kg_agent.research init --dir research-runs/demo --budget 3
uv run python -m kg_agent.research run --dir research-runs/demo
uv run python -m kg_agent.research status --dir research-runs/demo
uv run python -m kg_agent.research generate --dir research-runs/demo

cat > research-runs/demo/proposal.json <<'JSON'
{
  "id": "quadratic",
  "hypothesis": "A quadratic term will reduce validation MSE by at least 1%",
  "rationale": "Test whether curvature explains error left by the linear baseline",
  "degree": 2,
  "penalty": 0
}
JSON

uv run python -m kg_agent.research propose --dir research-runs/demo --file research-runs/demo/proposal.json
uv run python -m kg_agent.research run --dir research-runs/demo
uv run python -m kg_agent.research finalize --dir research-runs/demo
```

This walkthrough finalizes early with one unused slot. For a full cycle, inspect
the new result, propose and run another allowed configuration before finalizing.

## Experiment and contracts

The fixed benchmark has 80 training, 40 validation, and 40 test observations with
independent deterministic random streams. The target is a quadratic function with
Gaussian noise, and the input lies in `[-1, 1]`. This is an intentionally small
example for validating research mechanics.

The model minimizes training MSE plus `penalty * sum(coefficients[1:] ** 2)`.
Polynomial degree is an integer from 1 through 5; penalties are `0`, `0.001`,
`0.01`, `0.1`, and `1`. The intercept is unpenalized. Defaults are seed 42 and
five trials including the degree-1, zero-penalty baseline. Budgets may be 1–25.
Only validation MSE determines the winner, with earlier trials winning exact ties.

`ResearchOntology` owns typed relations, allowed configurations, command guards,
and fit/evaluate contracts. `plan_for` orders an evaluation after its experiment.
The executor supplies coefficients and finite metrics, checks the contracts,
and then records completion. Proposal prose never becomes measured evidence.

A proposal freezes its comparison target when submitted. The evaluator records
`observed_improvement` for at least 1% relative validation improvement, or
`no_observed_improvement` otherwise. A zero-MSE reference cannot be improved.
The final winner is the lowest validation MSE even if the margin is below 1%.

## Process and persistence contract

All operational commands output one JSON object:

```json
{"ok": true, "result": {}}
```

Errors use `{"ok": false, "error": {"code": "...", "message": "..."}}`.
Exit status is 0 for success, 2 for invalid requests, and 3 for experiment,
storage, invalid-state, or lock failures. `--help` prints conventional help text.
There are no interactive prompts. Integration requires only terminal commands;
no Codex- or Claude-specific runtime dependency is introduced.

Every command includes a top-level `description` explaining its purpose. Trial
objects returned by `status`, `propose`, and `run` include ordered `steps` with
`step`, `status`, and `description` fields. These describe the hypothesis, contract,
plan, fit, evaluation, comparison, and persistence, including actual saved metrics
and whether the 1% criterion was met. The status result also explains what to do next.
Descriptions are derived from saved evidence, not timestamped execution logs.
Pending and skipped actions are identified explicitly. `report.md` includes the
same step-by-step explanation and a description of finalization. For an existing
finalized session, rerun `finalize` to refresh the report without rerunning research.

- `state.json` is authoritative: versioned session settings, all trial records,
  coefficients, validation evidence, graph history, and the final report.
- `report.md` is generated upon finalization. Repeating `finalize` repairs a missing
  report from saved state without evaluating the test set again.
- Writes replace the state atomically. An interrupted trial reruns deterministically
  from its last committed state with the same ID and budget slot. Test evaluation
  is committed together with finalization; interruption before that commit can
  require recomputation, but cannot expose an intermediate score through the CLI.
- An OS process lock prevents concurrent CLI writers and releases on process death.
  This implementation targets POSIX systems (Linux/macOS). The session directory
  should be on a local filesystem supporting file locks and atomic rename.
- Completed trials cannot be rerun. Numerical failures consume a slot; a baseline
  failure blocks the study. A failed follow-up leaves the best successful model
  available for subsequent comparisons and finalization.
- Repeating an identical proposal ID is a readback, including after finalization;
  changing its contents is rejected. New proposals after finalization are rejected.
- Loading explicitly restores the research ontology; unknown state/ontology versions
  are rejected rather than silently migrated.

The Python `ResearchSession` API is also usable in tests and applications. Its
caller must hold `session_lock` for concurrent access; the CLI does this itself.

`generate` previews one deterministic proposal anchored on the current lowest
validation-MSE trial. It considers only unseen degree/penalty configurations,
prefers nearby one-factor changes, and does not evaluate data, consume budget,
queue a trial, or write session state. The returned proposal can be submitted
with `propose` after review.
