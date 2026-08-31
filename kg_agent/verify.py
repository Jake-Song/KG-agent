"""Constrain hallucination: check a claim against the graph before believing it.

Every statement a model produces is reduced to a :class:`Claim` and given a
:class:`Status` -- ``supported``, ``entailed``, ``unknown``, ``contradicted``
or ``ill_formed`` -- so the agent can stop treating all generations as equally
plausible.  :class:`IngestPolicy` then decides what is allowed into the world
model, which is what keeps the other two capabilities trustworthy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from .graph import Edge, KnowledgeGraph

if TYPE_CHECKING:  # pragma: no cover
    from .schema import Ontology

__all__ = ["Status", "Claim", "Verdict", "IngestPolicy", "IngestReport",
           "verify", "verify_all", "ingest"]


class Status(StrEnum):
    SUPPORTED = "supported"          # the edge is asserted in the graph
    ENTAILED = "entailed"            # derivable (inverse / symmetric / transitive)
    UNKNOWN = "unknown"              # no evidence either way
    CONTRADICTED = "contradicted"    # the graph asserts something incompatible
    ILL_FORMED = "ill_formed"        # unknown predicate/entity or a type violation


@dataclass(frozen=True, slots=True)
class Claim:
    """A model-generated statement, already reduced to a triple."""

    subject: str
    predicate: str
    object: str
    text: str | None = None
    source: str = "llm"
    confidence: float | None = None

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    @classmethod
    def parse(cls, text: str, **kwargs) -> Claim:
        """``Claim.parse("Protein_A inhibits Protein_B")``."""
        parts = text.replace("-->", " ").replace("--", " ").split()
        if len(parts) != 3:
            raise ValueError(f"expected 'subject predicate object', got {text!r}")
        return cls(parts[0], parts[1], parts[2], text=text, **kwargs)


@dataclass(slots=True)
class Verdict:
    claim: Claim
    status: Status
    evidence: list[Edge] = field(default_factory=list)
    derivation: list[Edge] | None = None
    explanation: str = ""

    @property
    def trusted(self) -> bool:
        return self.status in (Status.SUPPORTED, Status.ENTAILED)

    def __str__(self) -> str:
        return f"[{self.status}] {self.claim} -- {self.explanation}"


def verify(kg: KnowledgeGraph, claim: Claim, *, ontology: Ontology | None = None,
           require_known_entities: bool = True, max_depth: int = 8) -> Verdict:
    """Judge one claim against the graph.

    Checks run contradiction-first: a graph that somehow holds both the claimed
    edge and an incompatible one is reported as contradicted rather than
    supported, because the conservative answer is the useful one.
    """
    onto = ontology if ontology is not None else kg.ontology
    s, p, o = claim.triple

    errors = onto.type_errors(kg, s, p, o)
    if require_known_entities:
        errors += [f"unknown entity '{n}'" for n in (s, o) if not kg.has_node(n)]
    if errors:
        return Verdict(claim, Status.ILL_FORMED, explanation="; ".join(errors))

    conflicts = onto.conflicts(kg, s, p, o)
    if conflicts:
        return Verdict(claim, Status.CONTRADICTED, evidence=conflicts,
                       explanation="graph asserts " + "; ".join(str(e) for e in conflicts))

    direct = kg.get_edge(s, p, o)
    if direct is not None:
        return Verdict(claim, Status.SUPPORTED, evidence=[direct],
                       explanation=f"asserted at rev {direct.asserted_at} "
                                   f"(source={direct.source}, confidence={direct.confidence:.2f})")

    if onto.is_symmetric(p):
        mirror = kg.get_edge(o, p, s)
        if mirror is not None:
            return Verdict(claim, Status.ENTAILED, evidence=[mirror], derivation=[mirror],
                           explanation=f"'{p}' is symmetric and {mirror} holds")

    inverse = onto.inverse_of(p)
    if inverse:
        mirror = kg.get_edge(o, inverse, s)
        if mirror is not None:
            return Verdict(claim, Status.ENTAILED, evidence=[mirror], derivation=[mirror],
                           explanation=f"'{inverse}' is the inverse of '{p}' and {mirror} holds")

    if onto.is_transitive(p):
        chain = kg.path(s, o, [p], max_depth=max_depth)
        if chain:
            return Verdict(claim, Status.ENTAILED, evidence=list(chain), derivation=list(chain),
                           explanation=f"'{p}' is transitive along " +
                                       " -> ".join(e.subject for e in chain) + f" -> {o}")

    return Verdict(claim, Status.UNKNOWN,
                   explanation=f"no edge, inverse or derivation for {claim}")


def verify_all(kg: KnowledgeGraph, claims: Iterable[Claim], **kwargs) -> list[Verdict]:
    return [verify(kg, c, **kwargs) for c in claims]


@dataclass(slots=True)
class IngestReport:
    """What a batch of claims did to the world model."""

    accepted: list[Verdict] = field(default_factory=list)      # already known
    provisional: list[Verdict] = field(default_factory=list)   # newly written, low trust
    rejected: list[Verdict] = field(default_factory=list)
    written: list[Edge] = field(default_factory=list)

    @property
    def contradictions(self) -> list[Verdict]:
        return [v for v in self.rejected if v.status is Status.CONTRADICTED]

    @property
    def verdicts(self) -> list[Verdict]:
        return self.accepted + self.provisional + self.rejected

    def summary(self) -> str:
        return (f"{len(self.accepted)} known, {len(self.provisional)} provisional, "
                f"{len(self.rejected)} rejected "
                f"({len(self.contradictions)} contradicted)")


@dataclass(slots=True)
class IngestPolicy:
    """Gate between what a model says and what the world model records."""

    accept_unknown: bool = True
    unknown_confidence: float = 0.5
    unknown_source: str = "llm"
    allow_new_entities: bool = True
    promote_supported: bool = False   # raise stored confidence when re-observed

    def apply(self, kg: KnowledgeGraph, verdicts: Iterable[Verdict]) -> IngestReport:
        report = IngestReport()
        for verdict in verdicts:
            if verdict.status in (Status.SUPPORTED, Status.ENTAILED):
                report.accepted.append(verdict)
                if self.promote_supported and verdict.status is Status.SUPPORTED:
                    s, p, o = verdict.claim.triple
                    kg.assert_edge(s, p, o, confidence=1.0, source="observation")
            elif verdict.status is Status.UNKNOWN and self.accept_unknown:
                s, p, o = verdict.claim.triple
                confidence = verdict.claim.confidence or self.unknown_confidence
                result = kg.assert_edge(s, p, o, confidence=confidence,
                                        source=verdict.claim.source or self.unknown_source)
                if result and result.edge is not None:
                    report.provisional.append(verdict)
                    report.written.append(result.edge)
                    report.written.extend(result.entailed)
                else:
                    verdict.status = Status.CONTRADICTED
                    verdict.evidence = result.conflicts
                    verdict.explanation = "; ".join(result.errors) or verdict.explanation
                    report.rejected.append(verdict)
            else:
                report.rejected.append(verdict)
        return report


def ingest(kg: KnowledgeGraph, claims: Iterable[Claim], *,
           policy: IngestPolicy | None = None, **verify_kwargs) -> IngestReport:
    """Verify then conditionally write -- the full claim pipeline."""
    policy = policy or IngestPolicy()
    verify_kwargs.setdefault("require_known_entities", not policy.allow_new_entities)
    return policy.apply(kg, verify_all(kg, claims, **verify_kwargs))
