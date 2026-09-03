"""The loop: observe -> update KG -> query KG -> plan -> choose -> act -> observe.

The agent holds no message history.  Every model call is handed a freshly
rendered slice of the graph (:meth:`KnowledgeGraph.context_for`), so the
context window carries a *view of the current world*, not the whole past.  The
graph is the state that persists.

The graph plans; the model may pick which *ready* step goes first, but only a
step the planner already admits is ever taken.  Each turn acts on one step
(``execute="step"``) or on the whole ready frontier (``execute="stage"``).
When an observation contradicts the graph, the agent resolves it against
provenance instead of silently dropping it: a provisional model-sourced edge
gives way to a more confident observation; anything else is kept and reported.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .graph import Edge, KnowledgeGraph
from .llm import LLM
from .planner import DEFAULT_ACTIONS, Goal, Plan, PlanStep, plan_for
from .verify import IngestPolicy, IngestReport, Verdict, ingest

__all__ = ["ActionResult", "Resolution", "LoopRecord", "RunResult", "KGAgent"]

EXECUTE_MODES = ("step", "stage")


@dataclass(slots=True)
class ActionResult:
    step: PlanStep
    ok: bool
    observation: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Resolution:
    """What the agent did about one contradicted claim."""

    verdict: Verdict
    decision: str                       # "kept" | "replaced"
    reason: str
    retracted: list[Edge] = field(default_factory=list)   # evidence + its inference twins
    report: IngestReport | None = None  # the re-ingest of the claim, when replaced

    def __str__(self) -> str:
        return f"{self.decision}: {self.verdict.claim} -- {self.reason}"


@dataclass(slots=True)
class LoopRecord:
    """One acted step, kept for inspection -- not fed back to the model.

    In ``stage`` mode several records share a ``turn``: one plan, many steps.
    """

    iteration: int
    plan: Plan
    step: PlanStep | None = None
    result: ActionResult | None = None
    report: IngestReport | None = None
    turn: int = 0
    chosen_by: str = "planner"          # "model" when the LLM picked this step
    resolutions: list[Resolution] = field(default_factory=list)
    note: str = ""                      # set on turns that acted on nothing (unblocking)


@dataclass(slots=True)
class RunResult:
    goal: Goal
    records: list[LoopRecord] = field(default_factory=list)
    plan: Plan | None = None
    completed: bool = False
    reason: str = ""

    @property
    def contradictions(self) -> list[Verdict]:
        """Every contradiction observed during the run, resolved or not."""
        out: list[Verdict] = []
        for record in self.records:
            if record.report:
                out.extend(record.report.contradictions)
        return out

    @property
    def resolutions(self) -> list[Resolution]:
        return [r for record in self.records for r in record.resolutions]

    @property
    def unresolved(self) -> list[Verdict]:
        """Contradictions the agent kept the graph's side of."""
        return [r.verdict for r in self.resolutions if r.decision == "kept"]

    def render(self) -> str:
        lines = [f"goal: {self.goal}",
                 f"outcome: {'completed' if self.completed else 'stopped'} ({self.reason})"]
        per_turn: dict[int, int] = {}
        for record in self.records:
            if record.step is not None:
                per_turn[record.turn] = per_turn.get(record.turn, 0) + 1
        show_turns = any(n > 1 for n in per_turn.values())
        seen_turns: set[int] = set()
        for record in self.records:
            if record.step is None:
                if record.note:
                    lines.append(f"  turn {record.turn}: {record.note}")
                continue
            if show_turns and record.turn not in seen_turns:
                seen_turns.add(record.turn)
                ready = len(record.plan.ready)
                acted = per_turn[record.turn]
                size = f"frontier of {ready}" + (f", {acted} acted" if acted < ready else "")
                lines.append(f"  turn {record.turn}: {size}")
            mark = "ok" if record.result and record.result.ok else "failed"
            chosen = " (model's choice)" if record.chosen_by == "model" else ""
            lines.append(f"  {record.iteration}. {record.step} -> {mark}{chosen}")
            if record.result:
                lines.append(f"       observed: {record.result.observation}")
            if record.report:
                lines.append(f"       ingest: {record.report.summary()}")
            for resolution in record.resolutions:
                if resolution.decision == "replaced":
                    lines.append(f"       resolved: {resolution}")
        kept = [r for r in self.resolutions if r.decision == "kept"]
        if kept:
            lines.append("  contradictions needing resolution:")
            lines.extend(f"    - {r.verdict} (kept: {r.reason})" for r in kept)
        return "\n".join(lines)


Handler = Callable[["KGAgent", PlanStep], "ActionResult | str | bool | None"]
Resolver = Callable[["KGAgent", Verdict], "str | None"]   # "keep" | "replace" | None (= default rule)


class KGAgent:
    """An agent whose entire state is a :class:`KnowledgeGraph`."""

    def __init__(self, kg: KnowledgeGraph, llm: LLM, *,
                 policy: IngestPolicy | None = None,
                 actions: Mapping[str, str] | None = None,
                 context_hops: int = 2,
                 max_retries: int = 1,
                 execute: str = "step") -> None:
        if execute not in EXECUTE_MODES:
            raise ValueError(f"execute must be one of {EXECUTE_MODES}, got {execute!r}")
        self.kg = kg
        self.llm = llm
        self.policy = policy or IngestPolicy()
        self.action_map = dict(DEFAULT_ACTIONS if actions is None else actions)
        self.context_hops = context_hops
        self.max_retries = max_retries
        self.execute = execute
        self._handlers: dict[str, Handler] = {}
        self._resolver: Resolver | None = None
        self._failures: dict[str, int] = {}

    # --------------------------------------------------------------- actions

    def action(self, name: str) -> Callable[[Handler], Handler]:
        """Register a handler: ``@agent.action("run_measurement")``."""

        def decorator(fn: Handler) -> Handler:
            self._handlers[name] = fn
            return fn

        return decorator

    def register(self, name: str, handler: Handler) -> None:
        self._handlers[name] = handler

    def resolver(self, fn: Resolver) -> Resolver:
        """Register a contradiction hook: ``@agent.resolver``.

        Called with ``(agent, verdict)``; return ``"keep"``, ``"replace"`` or
        ``None`` to fall back to the default provenance rule.
        """
        self._resolver = fn
        return fn

    def on_contradiction(self, fn: Resolver) -> None:
        self._resolver = fn

    # ----------------------------------------------------------- the loop

    def context(self, focus: str | Iterable[str], hops: int | None = None) -> str:
        return self.kg.context_for(focus, hops=self.context_hops if hops is None else hops)

    def observe(self, observation: str, *, focus: str | None = None) -> IngestReport:
        """Extract claims from an observation, verify them, write what survives."""
        context = self.context(focus) if focus else ""
        proposals = list(self.llm.propose_claims(observation, context))
        return ingest(self.kg, proposals, policy=self.policy)

    def plan(self, goal: Goal | str) -> Plan:
        return plan_for(self.kg, goal, actions=self.action_map)

    def act(self, step: PlanStep) -> ActionResult:
        """Run the handler for a step and fold its outcome back into the graph."""
        handler = self._handlers.get(step.action)
        if handler is None:
            return ActionResult(step, False, f"no handler registered for '{step.action}'",
                                {"unhandled": True})
        outcome = handler(self, step)
        if isinstance(outcome, ActionResult):
            result = outcome
        elif isinstance(outcome, str):
            result = ActionResult(step, True, outcome)
        elif outcome is False:
            result = ActionResult(step, False, f"{step.action} on {step.node} failed")
        else:
            result = ActionResult(step, True, f"{step.action} on {step.node} succeeded")

        if result.ok:
            node = self.kg.get_node(step.node)
            if node is None or node.attrs.get("status") not in self.kg.ontology.satisfied_states:
                self.kg.update_node(step.node, status="complete")
        else:
            self._failures[step.node] = self._failures.get(step.node, 0) + 1
            self.kg.update_node(step.node, status="failed")
        return result

    def resolve(self, report: IngestReport) -> list[Resolution]:
        """Decide each contradiction in an ingest report; see :meth:`resolver`."""
        return [self._resolve(verdict) for verdict in report.contradictions]

    def step(self, goal: Goal | str, iteration: int = 0) -> LoopRecord:
        """One full cycle: query the graph, plan, choose, act, observe, resolve."""
        plan = self.plan(goal)
        if plan.complete or not plan.ready:
            return LoopRecord(iteration=iteration, plan=plan, turn=iteration)
        chosen, chosen_by = self._select(plan)
        return self._execute(chosen, plan, iteration=iteration, turn=iteration,
                             chosen_by=chosen_by)

    def run(self, goal: Goal | str, max_steps: int = 20) -> RunResult:
        """Loop until the goal is satisfied, the agent is stuck, or the budget runs out.

        ``max_steps`` bounds acted steps plus turns spent unblocking a leaf.
        """
        goal = goal if isinstance(goal, Goal) else Goal(goal)
        run = RunResult(goal=goal)
        spent = iteration = turn = 0
        while True:
            plan = self.plan(goal)
            run.plan = plan
            if plan.complete:
                run.completed = True
                run.reason = "goal satisfied"
                break
            if spent >= max_steps:
                run.reason = f"step budget ({max_steps}) exhausted"
                break
            turn += 1
            frontier = plan.ready
            if not frontier:
                # Everything ready is blocked: ask the model, let the graph vet it.
                stuck = next(s for s in plan.steps if s.blocked)
                spent += 1
                report = self._unblock(stuck)
                written = len(report.written) if report else 0
                run.records.append(LoopRecord(
                    iteration=iteration, plan=plan, turn=turn, report=report,
                    note=(f"blocked on {stuck.node}; asked the model, which added "
                          f"{written} edge{'s' if written != 1 else ''}"
                          if written else f"blocked on {stuck.node}; the model offered nothing usable")))
                if not written:
                    run.reason = f"no way to satisfy {stuck.node}"
                    break
                continue
            first, chosen_by = self._select(plan)
            if self.execute == "step":
                batch = [first]
            else:
                batch = [first, *(s for s in frontier if s is not first)]
            abandoned: PlanStep | None = None
            for step in batch:
                if spent >= max_steps:
                    break
                iteration += 1
                spent += 1
                record = self._execute(step, plan, iteration=iteration, turn=turn,
                                       chosen_by=chosen_by if step is first else "planner")
                run.records.append(record)
                if (record.result is not None and not record.result.ok
                        and self._failures.get(step.node, 0) > self.max_retries):
                    abandoned = step
                    break
            if abandoned is not None:
                run.reason = f"{abandoned.node} failed {self._failures[abandoned.node]} times"
                break
        return run

    # -------------------------------------------------------------- internal

    def _select(self, plan: Plan) -> tuple[PlanStep, str]:
        """Which ready step goes first: the model's pick if it is on the plan."""
        ready = plan.ready
        if len(ready) < 2:
            return ready[0], "planner"
        choice = self.llm.choose_action(plan, self.context([s.node for s in ready]))
        step = plan.find(choice)
        return (step, "model") if step is not None else (ready[0], "planner")

    def _execute(self, step: PlanStep, plan: Plan, *, iteration: int, turn: int,
                 chosen_by: str) -> LoopRecord:
        record = LoopRecord(iteration=iteration, plan=plan, step=step, turn=turn,
                            chosen_by=chosen_by)
        record.result = self.act(step)
        record.report = self.observe(record.result.observation, focus=step.node)
        record.resolutions = self.resolve(record.report)
        return record

    def _unblock(self, step: PlanStep) -> IngestReport | None:
        """Ask the model for sub-dependencies; the graph vets every proposal.

        Returns the ingest report (check ``report.written``), or ``None`` when
        the model proposed nothing.
        """
        proposals = list(self.llm.propose_dependencies(step.node, self.context(step.node)))
        if not proposals:
            return None
        return ingest(self.kg, proposals, policy=self.policy)

    def _resolve(self, verdict: Verdict) -> Resolution:
        decision = self._resolver(self, verdict) if self._resolver else None
        if decision is None:
            decision, reason = self._default_decision(verdict)
        else:
            reason = "decided by the resolver hook"
        if decision == "replace":
            return self._replace(verdict, reason)
        return Resolution(verdict, "kept", reason)

    def _default_decision(self, verdict: Verdict) -> tuple[str, str]:
        """Provenance rule: a more confident claim beats provisional model-sourced edges.

        Evidence written by ``inference`` (a materialised inverse or symmetric
        twin) counts as non-provisional, so ontologies that declare inverse
        predicates mutually incompatible resolve to ``keep``.
        """
        if not self.policy.accept_unknown:
            return "keep", "policy refuses unknown claims, so nothing could replace the evidence"
        if not verdict.evidence:
            return "keep", "no conflicting edges to weigh"
        confidence = verdict.claim.confidence or self.policy.unknown_confidence
        for edge in verdict.evidence:
            if edge.source != self.policy.unknown_source:
                return "keep", f"{edge} comes from '{edge.source}', not a provisional model claim"
            if edge.confidence >= confidence:
                return "keep", (f"{edge} (confidence {edge.confidence:.2f}) is at least as "
                                f"confident as the claim ({confidence:.2f})")
        return "replace", (f"every conflicting edge is provisional "
                           f"('{self.policy.unknown_source}') and below the claim's "
                           f"confidence {confidence:.2f}")

    def _replace(self, verdict: Verdict, reason: str) -> Resolution:
        retracted: list[Edge] = []
        for edge in verdict.evidence:
            gone = self.kg.retract_edge(*edge.triple, reason="superseded by observation")
            if gone is not None:
                retracted.append(gone)
            for s, p, o in self.kg.ontology.entailments(*edge.triple):
                twin = self.kg.get_edge(s, p, o)
                if twin is not None and twin.source == "inference":
                    gone = self.kg.retract_edge(s, p, o, reason="superseded by observation")
                    if gone is not None:
                        retracted.append(gone)
        report = ingest(self.kg, [verdict.claim], policy=self.policy)
        if not report.written:
            reason += f"; the claim itself was not written ({report.summary()})"
        return Resolution(verdict, "replaced", reason, retracted=retracted, report=report)
