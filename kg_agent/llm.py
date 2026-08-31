"""The model boundary.

Nothing in this package talks to a provider.  An LLM is anything satisfying
:class:`LLM`: it proposes claims and candidate dependencies, and every
proposal is verified against the graph before it is believed.  :class:`ScriptedLLM`
is a deterministic stand-in used by the demo and the tests.

To use a real model, implement the same three methods -- prompt it with the
``context`` string (which comes from :meth:`KnowledgeGraph.context_for`, not
from a transcript) and parse its output into :class:`Claim` objects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .verify import Claim

if TYPE_CHECKING:  # pragma: no cover
    from .planner import Plan

__all__ = ["LLM", "ScriptedLLM", "claims"]


@runtime_checkable
class LLM(Protocol):
    """What the agent needs from a model."""

    def propose_claims(self, observation: str, context: str) -> Sequence[Claim]:
        """Extract triples from an observation, given the world-model context."""

    def propose_dependencies(self, node: str, context: str) -> Sequence[Claim]:
        """Suggest what ``node`` might require, when planning dead-ends."""

    def choose_action(self, plan: Plan, context: str) -> str | None:
        """Optionally override the planner's next action by name."""


def claims(*specs: str, source: str = "llm", confidence: float | None = None) -> list[Claim]:
    """``claims("Protein_A inhibits Protein_B", ...)`` -- terse test/demo helper."""
    return [Claim.parse(s, source=source, confidence=confidence) for s in specs]


class ScriptedLLM:
    """Deterministic :class:`LLM` driven by lookup tables.

    Keys are matched exactly, then by substring, so demo observations can be
    phrased naturally without brittle exact-match tables.
    """

    def __init__(self,
                 claims_by_observation: Mapping[str, Iterable[Claim]] | None = None,
                 dependencies_by_node: Mapping[str, Iterable[Claim]] | None = None,
                 action: Callable[[Plan, str], str | None] | None = None) -> None:
        self.claims_by_observation = {k: list(v) for k, v in (claims_by_observation or {}).items()}
        self.dependencies_by_node = {k: list(v) for k, v in (dependencies_by_node or {}).items()}
        self._action = action
        self.calls: list[tuple[str, str, str]] = []   # (method, key, context) for assertions

    def _lookup(self, table: Mapping[str, list[Claim]], key: str) -> list[Claim]:
        if key in table:
            return list(table[key])
        for pattern, value in table.items():
            if pattern in key:
                return list(value)
        return []

    def propose_claims(self, observation: str, context: str) -> list[Claim]:
        self.calls.append(("propose_claims", observation, context))
        return self._lookup(self.claims_by_observation, observation)

    def propose_dependencies(self, node: str, context: str) -> list[Claim]:
        self.calls.append(("propose_dependencies", node, context))
        return self._lookup(self.dependencies_by_node, node)

    def choose_action(self, plan: Plan, context: str) -> str | None:
        self.calls.append(("choose_action", str(plan), context))
        return self._action(plan, context) if self._action else None
