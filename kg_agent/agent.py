"""The loop: observe -> update KG -> query KG -> plan -> act -> observe.

The agent holds no message history.  Every model call is handed a freshly
rendered slice of the graph (:meth:`KnowledgeGraph.context_for`), so the
context window carries a *view of the current world*, not the whole past.  The
graph is the state that persists.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .graph import KnowledgeGraph
from .llm import LLM
from .planner import DEFAULT_ACTIONS, Goal, Plan, PlanStep, plan_for
from .verify import IngestPolicy, IngestReport, Verdict, ingest

__all__ = ["ActionResult", "LoopRecord", "RunResult", "KGAgent"]


@dataclass(slots=True)
class ActionResult:
    step: PlanStep
    ok: bool
    observation: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoopRecord:
    """One turn of the loop, kept for inspection -- not fed back to the model."""

    iteration: int
    plan: Plan
    step: PlanStep | None = None
    result: ActionResult | None = None
    report: IngestReport | None = None


@dataclass(slots=True)
class RunResult:
    goal: Goal
    records: list[LoopRecord] = field(default_factory=list)
    plan: Plan | None = None
    completed: bool = False
    reason: str = ""

    @property
    def contradictions(self) -> list[Verdict]:
        out: list[Verdict] = []
        for record in self.records:
            if record.report:
                out.extend(record.report.contradictions)
        return out

    def render(self) -> str:
        lines = [f"goal: {self.goal}",
                 f"outcome: {'completed' if self.completed else 'stopped'} ({self.reason})"]
        for record in self.records:
            if record.step is None:
                continue
            mark = "ok" if record.result and record.result.ok else "failed"
            lines.append(f"  {record.iteration}. {record.step} -> {mark}")
            if record.result:
                lines.append(f"       observed: {record.result.observation}")
            if record.report:
                lines.append(f"       ingest: {record.report.summary()}")
        if self.contradictions:
            lines.append("  contradictions needing resolution:")
            lines.extend(f"    - {v}" for v in self.contradictions)
        return "\n".join(lines)


Handler = Callable[["KGAgent", PlanStep], "ActionResult | str | bool | None"]


class KGAgent:
    """An agent whose entire state is a :class:`KnowledgeGraph`."""

    def __init__(self, kg: KnowledgeGraph, llm: LLM, *,
                 policy: IngestPolicy | None = None,
                 actions: Mapping[str, str] | None = None,
                 context_hops: int = 2,
                 max_retries: int = 1) -> None:
        self.kg = kg
        self.llm = llm
        self.policy = policy or IngestPolicy()
        self.action_map = dict(DEFAULT_ACTIONS if actions is None else actions)
        self.context_hops = context_hops
        self.max_retries = max_retries
        self._handlers: dict[str, Handler] = {}
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

    def step(self, goal: Goal | str, iteration: int = 0) -> LoopRecord:
        """One full cycle: query the graph, plan, act, observe the result."""
        plan = self.plan(goal)
        record = LoopRecord(iteration=iteration, plan=plan)
        next_step = plan.next_step()
        if next_step is None:
            return record
        record.step = next_step
        record.result = self.act(next_step)
        record.report = self.observe(record.result.observation, focus=next_step.node)
        return record

    def run(self, goal: Goal | str, max_steps: int = 20) -> RunResult:
        goal = goal if isinstance(goal, Goal) else Goal(goal)
        run = RunResult(goal=goal)
        for iteration in range(1, max_steps + 1):
            plan = self.plan(goal)
            run.plan = plan
            if plan.complete:
                run.completed = True
                run.reason = "goal satisfied"
                break
            next_step = plan.next_step()
            assert next_step is not None
            if next_step.blocked:
                if not self._unblock(next_step):
                    run.records.append(LoopRecord(iteration=iteration, plan=plan))
                    run.reason = f"no way to satisfy {next_step.node}"
                    break
                continue
            record = self.step(goal, iteration=iteration)
            run.records.append(record)
            run.plan = record.plan
            if record.result is not None and not record.result.ok:
                if self._failures.get(next_step.node, 0) > self.max_retries:
                    run.reason = f"{next_step.node} failed {self._failures[next_step.node]} times"
                    break
        else:
            run.reason = f"step budget ({max_steps}) exhausted"
        if not run.completed and not run.reason:
            run.reason = "stopped"
        return run

    # -------------------------------------------------------------- internal

    def _unblock(self, step: PlanStep) -> bool:
        """Ask the model for sub-dependencies; the graph vets every proposal."""
        proposals = list(self.llm.propose_dependencies(step.node, self.context(step.node)))
        if not proposals:
            return False
        report = ingest(self.kg, proposals, policy=self.policy)
        return bool(report.written)
