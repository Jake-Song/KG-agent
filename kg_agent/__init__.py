"""A knowledge graph used as an agent's world model, planner and truth filter.

Three capabilities over one graph:

1. **World model** -- :class:`KGAgent` runs observe -> update -> query -> plan ->
   act -> observe, keeping no transcript; state lives in :class:`KnowledgeGraph`.
2. **Planning** -- :func:`plan_for` backward-chains over dependency edges.
3. **Hallucination constraint** -- :func:`verify` labels every model claim
   supported / entailed / unknown / contradicted / ill_formed before
   :class:`IngestPolicy` decides what enters the graph.
"""

from .agent import ActionResult, KGAgent, LoopRecord, RunResult
from .graph import AssertResult, ConstraintViolation, Edge, KnowledgeGraph, Node
from .llm import LLM, ScriptedLLM, claims
from .planner import (DEFAULT_ACTIONS, CyclicDependencyError, Goal, Plan, PlanStep,
                      plan_for)
from .schema import Ontology, RelationSpec, default_ontology
from .verify import (Claim, IngestPolicy, IngestReport, Status, Verdict, ingest,
                     verify, verify_all)

__all__ = [
    "ActionResult", "AssertResult", "Claim", "ConstraintViolation", "CyclicDependencyError",
    "DEFAULT_ACTIONS", "DEFAULT_MODEL", "Edge", "Goal", "IngestPolicy", "IngestReport", "KGAgent",
    "KnowledgeGraph", "LLM", "LoopRecord", "Node", "Ontology", "OpenRouterError",
    "OpenRouterLLM", "Plan", "PlanStep",
    "RelationSpec", "RunResult", "ScriptedLLM", "Status", "Usage", "Verdict", "claims",
    "default_ontology", "ingest", "plan_for", "verify", "verify_all",
]
__version__ = "0.1.0"


# Imported lazily so `python -m kg_agent.openrouter` does not double-import the module.
_OPENROUTER = {"DEFAULT_MODEL", "OpenRouterError", "OpenRouterLLM", "Usage"}


def __getattr__(name: str):
    if name in _OPENROUTER:
        from . import openrouter

        return getattr(openrouter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
