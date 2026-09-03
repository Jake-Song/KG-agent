"""Turn dependency edges into an ordered plan.

The graph doubles as a symbolic planner's state representation: a goal is
resolved by backward-chaining over the ontology's dependency predicates
(``requires``, ``depends_on``, ``measured_by``), pruning anything the world
model already records as satisfied.  A model can propose candidate steps and
pick which *ready* step goes first, but the graph decides which are admissible
and which stage they belong to.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .graph import KnowledgeGraph

if TYPE_CHECKING:  # pragma: no cover
    from .schema import Ontology

__all__ = ["Goal", "PlanStep", "Plan", "plan_for", "CyclicDependencyError",
           "DEFAULT_ACTIONS", "DEFAULT_ACTION"]

DEFAULT_ACTION = "resolve"

#: node type -> the action that satisfies a node of that type
DEFAULT_ACTIONS: dict[str, str] = {
    "Hypothesis": "evaluate_hypothesis",
    "Measurement": "run_measurement",
    "Instrument": "acquire_instrument",
    "Lab": "secure_lab_access",
    "Dataset": "load_dataset",
    "Experiment": "run_experiment",
    "Assumption": "check_assumption",
}


class CyclicDependencyError(Exception):
    """The dependency subgraph reachable from the goal contains a cycle."""

    def __init__(self, cycle: list[str]) -> None:
        super().__init__("cyclic dependency: " + " -> ".join(cycle))
        self.cycle = cycle


@dataclass(frozen=True, slots=True)
class Goal:
    target: str
    description: str = ""

    def __str__(self) -> str:
        return self.description or f"satisfy {self.target}"


@dataclass(frozen=True, slots=True)
class PlanStep:
    node: str
    node_type: str
    action: str
    depends_on: tuple[str, ...] = ()
    stage: int = 0
    blocked: bool = False

    def __str__(self) -> str:
        suffix = " [blocked: no known action]" if self.blocked else ""
        return f"{self.action}({self.node}){suffix}"


@dataclass(slots=True)
class Plan:
    goal: Goal
    steps: list[PlanStep] = field(default_factory=list)
    stages: list[list[PlanStep]] = field(default_factory=list)
    satisfied: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Nothing left to do -- the goal is already satisfied."""
        return not self.steps

    def next_step(self) -> PlanStep | None:
        return self.steps[0] if self.steps else None

    @property
    def ready(self) -> list[PlanStep]:
        """The actionable frontier: stage-0 steps (no pending dependency) that are not blocked."""
        return [s for s in (self.stages[0] if self.stages else []) if not s.blocked]

    def find(self, choice: str | None) -> PlanStep | None:
        """Resolve a model's reply to a ready step, or ``None``.

        Accepts the exact ``str(step)`` form (``upgrade_package(pluggy)``) first,
        then a bare action name, matched against the first ready step that has
        it.  Only :attr:`ready` steps qualify, so a model can never pick a step
        whose dependencies are unmet.
        """
        if not choice:
            return None
        choice = choice.strip()
        ready = self.ready
        for step in ready:
            if str(step) == choice:
                return step
        for step in ready:
            if step.action == choice:
                return step
        return None

    def render(self) -> str:
        lines = [f"goal: {self.goal}"]
        if self.satisfied:
            lines.append(f"  already satisfied: {', '.join(sorted(self.satisfied))}")
        if not self.steps:
            lines.append("  (nothing to do)")
        for i, stage in enumerate(self.stages):
            names = ", ".join(str(s) for s in stage)
            lines.append(f"  stage {i}: {names}")
        if self.blocked:
            lines.append(f"  blocked on: {', '.join(sorted(self.blocked))}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return " -> ".join(step.node for step in self.steps)


def plan_for(kg: KnowledgeGraph, goal: Goal | str, *, ontology: Ontology | None = None,
             actions: Mapping[str, str] | None = None,
             satisfied_states: Iterable[str] | None = None,
             max_depth: int = 32) -> Plan:
    """Backward-chain from ``goal`` to an ordered, stage-grouped plan."""
    onto = ontology if ontology is not None else kg.ontology
    goal = goal if isinstance(goal, Goal) else Goal(goal)
    action_map = DEFAULT_ACTIONS if actions is None else dict(actions)
    states = frozenset(satisfied_states) if satisfied_states is not None else onto.satisfied_states
    dependency_predicates = onto.dependency_predicates

    def is_satisfied(node: str) -> bool:
        entity = kg.get_node(node)
        if entity is not None and entity.attrs.get("status") in states:
            return True
        return any(kg.match(node, "satisfied_by", None))

    dependencies: dict[str, set[str]] = {}
    satisfied: set[str] = set()
    GRAY, BLACK = 1, 2
    color: dict[str, int] = {}

    def visit(node: str, trail: list[str]) -> None:
        state = color.get(node)
        if state == GRAY:
            raise CyclicDependencyError(trail[trail.index(node):] + [node])
        if state == BLACK:
            return
        color[node] = GRAY
        if is_satisfied(node):
            satisfied.add(node)
            color[node] = BLACK
            return
        children = sorted({e.object for e in kg.match(node, None, None)
                           if e.predicate in dependency_predicates})
        pending: set[str] = set()
        if len(trail) < max_depth:
            for child in children:
                visit(child, trail + [node])
                if child not in satisfied:
                    pending.add(child)
        dependencies[node] = pending
        color[node] = BLACK

    visit(goal.target, [])

    if goal.target in satisfied:
        return Plan(goal=goal, satisfied=sorted(satisfied))

    # Kahn's algorithm; ties broken alphabetically so plans are reproducible.
    stage_of: dict[str, int] = {}
    remaining = {n: set(deps) for n, deps in dependencies.items()}
    ready = sorted(n for n, deps in remaining.items() if not deps)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        stage_of[node] = max((stage_of[d] + 1 for d in dependencies[node]), default=0)
        newly: list[str] = []
        for other, deps in remaining.items():
            if node in deps:
                deps.discard(node)
                if not deps and other not in stage_of and other not in ready:
                    newly.append(other)
        ready = sorted(ready + newly)
    if len(order) != len(dependencies):  # pragma: no cover - visit() catches cycles first
        raise CyclicDependencyError(sorted(set(dependencies) - set(order)))

    steps: list[PlanStep] = []
    blocked: list[str] = []
    for node in order:
        node_type = kg.node_type(node) or "Unknown"
        deps = tuple(sorted(dependencies[node]))
        is_blocked = not deps and node_type not in action_map
        if is_blocked:
            blocked.append(node)
        steps.append(PlanStep(node=node, node_type=node_type,
                              action=action_map.get(node_type, DEFAULT_ACTION),
                              depends_on=deps, stage=stage_of[node], blocked=is_blocked))

    steps.sort(key=lambda s: (s.stage, s.node))
    stages: list[list[PlanStep]] = []
    for step in steps:
        while len(stages) <= step.stage:
            stages.append([])
        stages[step.stage].append(step)

    return Plan(goal=goal, steps=steps, stages=stages, satisfied=sorted(satisfied),
                blocked=blocked)
