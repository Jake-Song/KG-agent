"""Relation semantics.

A bare triple store cannot tell you that ``inhibits`` contradicts
``activates`` -- contradiction is undetectable without declared semantics.
This module supplies that layer: which relations are functional, symmetric,
transitive, mutually exclusive, and which node types they connect.  It is what
makes the ``contradicted`` verdict and the dependency planner possible.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .graph import Edge, KnowledgeGraph

__all__ = ["RelationSpec", "Ontology", "default_ontology"]

UNKNOWN_TYPE = "Unknown"


@dataclass(frozen=True, slots=True)
class RelationSpec:
    """Declared meaning of one predicate."""

    name: str
    inverse: str | None = None
    symmetric: bool = False
    functional: bool = False          # at most one object per subject
    transitive: bool = False          # a--p-->b--p-->c entails a--p-->c
    incompatible_with: frozenset[str] = frozenset()
    domain: str | None = None         # required subject type
    range: str | None = None          # required object type
    dependency: bool = False          # planner traverses this edge
    description: str = ""


class Ontology:
    """A set of :class:`RelationSpec` with the closure rules applied."""

    def __init__(self, relations: Iterable[RelationSpec] = (), *,
                 strict_predicates: bool = True,
                 satisfied_states: Iterable[str] = ("complete", "available", "satisfied",
                                                    "validated", "done")) -> None:
        self.relations: dict[str, RelationSpec] = {r.name: r for r in relations}
        self.strict_predicates = strict_predicates
        self.satisfied_states = frozenset(satisfied_states)
        self._incompatible: dict[str, set[str]] = {}
        self._inverse: dict[str, str] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        """Make incompatibility symmetric and inverses mutual."""
        self._incompatible = {name: set(spec.incompatible_with)
                              for name, spec in self.relations.items()}
        for name, others in list(self._incompatible.items()):
            for other in others:
                self._incompatible.setdefault(other, set()).add(name)
        self._inverse = {}
        for name, spec in self.relations.items():
            if spec.inverse:
                self._inverse[name] = spec.inverse
                self._inverse.setdefault(spec.inverse, name)

    # ------------------------------------------------------------- accessors

    def add(self, spec: RelationSpec) -> None:
        self.relations[spec.name] = spec
        self._rebuild()

    def get(self, predicate: str) -> RelationSpec | None:
        return self.relations.get(predicate)

    def knows(self, predicate: str) -> bool:
        return predicate in self.relations

    def is_functional(self, predicate: str) -> bool:
        spec = self.relations.get(predicate)
        return bool(spec and spec.functional)

    def is_transitive(self, predicate: str) -> bool:
        spec = self.relations.get(predicate)
        return bool(spec and spec.transitive)

    def is_symmetric(self, predicate: str) -> bool:
        spec = self.relations.get(predicate)
        return bool(spec and spec.symmetric)

    def inverse_of(self, predicate: str) -> str | None:
        return self._inverse.get(predicate)

    def incompatible(self, predicate: str) -> frozenset[str]:
        return frozenset(self._incompatible.get(predicate, ()))

    def domain_of(self, predicate: str) -> str | None:
        spec = self.relations.get(predicate)
        return spec.domain if spec else None

    def range_of(self, predicate: str) -> str | None:
        spec = self.relations.get(predicate)
        return spec.range if spec else None

    @property
    def dependency_predicates(self) -> frozenset[str]:
        """Predicates the planner backward-chains over."""
        return frozenset(name for name, spec in self.relations.items() if spec.dependency)

    # ------------------------------------------------------------- reasoning

    def type_errors(self, kg: KnowledgeGraph, subject: str, predicate: str,
                    object: str) -> list[str]:
        """Predicate/type problems that make a triple ill-formed."""
        errors: list[str] = []
        spec = self.relations.get(predicate)
        if spec is None:
            if self.strict_predicates and self.relations:
                errors.append(f"unknown predicate '{predicate}'")
            return errors
        for node_id, expected, role in ((subject, spec.domain, "subject"),
                                        (object, spec.range, "object")):
            if expected is None:
                continue
            actual = kg.node_type(node_id)
            if actual is not None and actual != UNKNOWN_TYPE and actual != expected:
                errors.append(
                    f"{role} '{node_id}' is a {actual}, but '{predicate}' requires {expected}")
        return errors

    def conflicts(self, kg: KnowledgeGraph, subject: str, predicate: str, object: str,
                  *, functional: bool = True) -> list[Edge]:
        """Live edges that contradict the proposed triple.

        ``functional=False`` skips functional rebinding, which is a state
        *update* when it comes from an observation but a *contradiction* when
        it comes from a model's claim.
        """
        found: list[Edge] = []
        for other in self.incompatible(predicate):
            found.extend(kg.match(subject, other, object))
            if self.is_symmetric(predicate) or self.is_symmetric(other):
                found.extend(kg.match(object, other, subject))
        inverse = self.inverse_of(predicate)
        if inverse:
            for other in self.incompatible(inverse):
                found.extend(kg.match(object, other, subject))
        if functional and self.is_functional(predicate):
            found.extend(e for e in kg.match(subject, predicate, None) if e.object != object)
        seen: set[tuple[str, str, str]] = set()
        unique: list[Edge] = []
        for edge in found:
            if edge.triple not in seen:
                seen.add(edge.triple)
                unique.append(edge)
        return unique

    def entailments(self, subject: str, predicate: str,
                    object: str) -> list[tuple[str, str, str]]:
        """Triples materialised alongside an assertion (inverse / symmetric)."""
        out: list[tuple[str, str, str]] = []
        if self.is_symmetric(predicate):
            out.append((object, predicate, subject))
        inverse = self.inverse_of(predicate)
        if inverse and inverse != predicate:
            out.append((object, inverse, subject))
        return out


def default_ontology(**kwargs: Any) -> Ontology:
    """The scientific-agent vocabulary used by the demo.

    Nothing in the core depends on it -- build your own :class:`Ontology` for
    another domain.
    """
    return Ontology(
        [
            RelationSpec("uses_dataset", domain=None, range="Dataset",
                         description="an experiment consumes a dataset"),
            RelationSpec("tests", range="Hypothesis",
                         description="an experiment tests a hypothesis"),
            RelationSpec("produced", range="Result",
                         description="an experiment produced a result"),
            RelationSpec("requires", transitive=True, dependency=True,
                         description="the subject cannot proceed without the object"),
            RelationSpec("depends_on", transitive=True, dependency=True,
                         description="weaker dependency, also planned over"),
            RelationSpec("measured_by", dependency=True, range="Measurement",
                         description="a hypothesis is settled by a measurement"),
            RelationSpec("located_at", functional=True, range="Lab",
                         description="an instrument sits in exactly one lab"),
            RelationSpec("conflicts_with", symmetric=True,
                         description="the two cannot both hold"),
            RelationSpec("satisfied_by",
                         description="a requirement already met by something concrete"),
            RelationSpec("supports", incompatible_with=frozenset({"refutes"}),
                         description="evidence for"),
            RelationSpec("refutes", incompatible_with=frozenset({"supports"}),
                         description="evidence against"),
            RelationSpec("inhibits", inverse="inhibited_by", domain="Protein", range="Protein",
                         incompatible_with=frozenset({"activates"})),
            RelationSpec("activates", inverse="activated_by", domain="Protein", range="Protein",
                         incompatible_with=frozenset({"inhibits"})),
            RelationSpec("inhibited_by", inverse="inhibits", domain="Protein", range="Protein"),
            RelationSpec("activated_by", inverse="activates", domain="Protein", range="Protein"),
        ],
        **kwargs,
    )
