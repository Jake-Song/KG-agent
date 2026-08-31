"""Indexed triple store that doubles as an agent's persistent world model.

The graph is not a question-answering cache: it *is* the agent's state.  Every
assertion carries provenance (source, confidence, revision) and every mutation
is journalled, so the environment can be reconstructed without replaying an
LLM conversation.  :meth:`KnowledgeGraph.context_for` renders a bounded
neighbourhood as text -- that rendering is what the agent hands to a model in
place of an ever-growing message history.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = ["Node", "Edge", "AssertResult", "KnowledgeGraph"]


@dataclass(slots=True)
class Node:
    """An entity.  ``attrs`` holds mutable state such as ``status="failed"``."""

    id: str
    type: str = "Unknown"
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "attrs": dict(self.attrs)}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Node:
        return cls(id=raw["id"], type=raw.get("type", "Unknown"), attrs=dict(raw.get("attrs", {})))


@dataclass(frozen=True, slots=True)
class Edge:
    """A provenance-carrying triple.  Retraction is soft: history survives."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "observation"
    asserted_at: int = 0
    retracted_at: int | None = None

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    @property
    def live(self) -> bool:
        return self.retracted_at is None

    def __str__(self) -> str:
        arrow = f"{self.subject} --{self.predicate}--> {self.object}"
        if not self.live:
            arrow += " (retracted)"
        return arrow

    def to_json(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "asserted_at": self.asserted_at,
            "retracted_at": self.retracted_at,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Edge:
        return cls(
            subject=raw["subject"],
            predicate=raw["predicate"],
            object=raw["object"],
            confidence=raw.get("confidence", 1.0),
            source=raw.get("source", "observation"),
            asserted_at=raw.get("asserted_at", 0),
            retracted_at=raw.get("retracted_at"),
        )


@dataclass(slots=True)
class AssertResult:
    """Outcome of :meth:`KnowledgeGraph.assert_edge`.

    ``status`` is one of ``added``, ``exists``, ``updated``, ``replaced`` (a
    functional relation was rebound) or ``rejected``.
    """

    status: str
    edge: Edge | None = None
    conflicts: list[Edge] = field(default_factory=list)
    entailed: list[Edge] = field(default_factory=list)
    retracted: list[Edge] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.status != "rejected"


class ConstraintViolation(Exception):
    """Raised by ``assert_edge(..., strict=True)`` when the ontology refuses."""

    def __init__(self, result: AssertResult) -> None:
        detail = "; ".join(result.errors) or "; ".join(str(e) for e in result.conflicts)
        super().__init__(f"assertion rejected: {detail}")
        self.result = result


class KnowledgeGraph:
    """Triple store with SPO/POS/OSP indexes, provenance and a change journal."""

    def __init__(self, ontology: Any | None = None) -> None:
        from .schema import Ontology  # local import: schema imports Edge from here

        self.ontology = ontology if ontology is not None else Ontology()
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, str], Edge] = {}
        self._history: list[Edge] = []
        self._spo: dict[str, dict[str, set[str]]] = {}
        self._pos: dict[str, dict[str, set[str]]] = {}
        self._osp: dict[str, dict[str, set[str]]] = {}
        self.journal: list[dict[str, Any]] = []
        self.revision = 0

    # ------------------------------------------------------------------ nodes

    def add_node(self, node_id: str, node_type: str = "Unknown", **attrs: Any) -> Node:
        node = self._nodes.get(node_id)
        if node is None:
            node = Node(id=node_id, type=node_type, attrs=dict(attrs))
            self._nodes[node_id] = node
            self._log("add_node", id=node_id, type=node_type, attrs=dict(attrs))
            return node
        if node_type != "Unknown" and node.type == "Unknown":
            node.type = node_type
            self._log("retype_node", id=node_id, type=node_type)
        if attrs:
            self.update_node(node_id, **attrs)
        return node

    def update_node(self, node_id: str, **attrs: Any) -> Node:
        node = self._nodes.get(node_id)
        if node is None:
            return self.add_node(node_id, **attrs)
        changed = {k: v for k, v in attrs.items() if node.attrs.get(k) != v}
        if changed:
            before = {k: node.attrs.get(k) for k in changed}
            node.attrs.update(changed)
            self._log("update_node", id=node_id, before=before, after=changed)
        return node

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node_type(self, node_id: str) -> str | None:
        node = self._nodes.get(node_id)
        return node.type if node else None

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    # ------------------------------------------------------------------ edges

    def assert_edge(
        self,
        subject: str,
        predicate: str,
        object: str,
        *,
        confidence: float = 1.0,
        source: str = "observation",
        strict: bool = False,
        _entailed: bool = False,
    ) -> AssertResult:
        """Assert a triple, enforcing the ontology.

        Rejects mutually-exclusive contradictions, rebinds functional
        relations (an *observation* updates state, where the verifier would
        call the same shape a contradiction), and materialises inverse and
        symmetric edges.
        """
        onto = self.ontology
        errors = onto.type_errors(self, subject, predicate, object)
        if errors:
            result = AssertResult("rejected", errors=errors)
            if strict:
                raise ConstraintViolation(result)
            return result

        conflicts = onto.conflicts(self, subject, predicate, object, functional=False)
        if conflicts:
            result = AssertResult("rejected", conflicts=conflicts,
                                  errors=[f"conflicts with {c}" for c in conflicts])
            if strict:
                raise ConstraintViolation(result)
            return result

        self._ensure_node(subject, onto.domain_of(predicate))
        self._ensure_node(object, onto.range_of(predicate))

        retracted: list[Edge] = []
        if onto.is_functional(predicate):
            for old in list(self.match(subject, predicate, None)):
                if old.object != object:
                    self.retract_edge(*old.triple, reason="functional rebinding")
                    retracted.append(old)

        key = (subject, predicate, object)
        existing = self._edges.get(key)
        if existing is not None and existing.live:
            if existing.confidence >= confidence and existing.source == source:
                return AssertResult("exists", edge=existing, retracted=retracted)
            updated = replace(existing, confidence=max(existing.confidence, confidence),
                              source=source if confidence >= existing.confidence else existing.source)
            self._history.append(existing)
            self._edges[key] = updated
            self._log("update_edge", **updated.to_json())
            return AssertResult("updated", edge=updated, retracted=retracted)

        self.revision += 1
        edge = Edge(subject, predicate, object, confidence=confidence, source=source,
                    asserted_at=self.revision)
        if existing is not None:
            self._history.append(existing)
        self._edges[key] = edge
        self._index(edge)
        self._log("assert_edge", **edge.to_json())

        entailed: list[Edge] = []
        if not _entailed:
            for s, p, o in onto.entailments(subject, predicate, object):
                sub = self.assert_edge(s, p, o, confidence=confidence, source="inference",
                                       _entailed=True)
                if sub.edge is not None and sub.status in {"added", "updated"}:
                    entailed.append(sub.edge)

        status = "replaced" if retracted else "added"
        return AssertResult(status, edge=edge, entailed=entailed, retracted=retracted)

    def retract_edge(self, subject: str, predicate: str, object: str,
                     *, reason: str = "") -> Edge | None:
        key = (subject, predicate, object)
        edge = self._edges.get(key)
        if edge is None or not edge.live:
            return None
        self.revision += 1
        retired = replace(edge, retracted_at=self.revision)
        self._edges[key] = retired
        self._unindex(edge)
        self._log("retract_edge", reason=reason, **retired.to_json())
        return retired

    def has_edge(self, subject: str, predicate: str, object: str) -> bool:
        edge = self._edges.get((subject, predicate, object))
        return edge is not None and edge.live

    def get_edge(self, subject: str, predicate: str, object: str) -> Edge | None:
        edge = self._edges.get((subject, predicate, object))
        return edge if edge is not None and edge.live else None

    @property
    def edges(self) -> list[Edge]:
        return [e for e in self._edges.values() if e.live]

    @property
    def history(self) -> list[Edge]:
        """Superseded and retracted edges, oldest first."""
        return self._history + [e for e in self._edges.values() if not e.live]

    # ---------------------------------------------------------------- queries

    def match(self, subject: str | None = None, predicate: str | None = None,
              object: str | None = None) -> Iterator[Edge]:
        """Triple-pattern query; ``None`` is a wildcard.  Uses the best index."""
        if subject is not None:
            for p, objects in self._pairs(self._spo.get(subject, {}), predicate):
                for o in sorted(objects):
                    if object is None or o == object:
                        yield self._edges[(subject, p, o)]
        elif object is not None:
            for s, predicates in self._pairs(self._osp.get(object, {}), None):
                for p in sorted(predicates):
                    if predicate is None or p == predicate:
                        yield self._edges[(s, p, object)]
        elif predicate is not None:
            for s, objects in self._pairs(self._pos.get(predicate, {}), None):
                for o in sorted(objects):
                    yield self._edges[(s, predicate, o)]
        else:
            for edge in sorted(self._edges.values(), key=lambda e: e.triple):
                if edge.live:
                    yield edge

    @staticmethod
    def _pairs(index: dict[str, set[str]], key: str | None):
        if key is None:
            return sorted(index.items())
        if key in index:
            return [(key, index[key])]
        return []

    def neighbors(self, subject: str, predicate: str | None = None) -> list[str]:
        return [e.object for e in self.match(subject, predicate, None)]

    def subjects(self, predicate: str, object: str) -> list[str]:
        return [e.subject for e in self.match(None, predicate, object)]

    def out_edges(self, subject: str) -> list[Edge]:
        return list(self.match(subject, None, None))

    def in_edges(self, object: str) -> list[Edge]:
        return list(self.match(None, None, object))

    def path(self, start: str, target: str, predicates: Iterable[str],
             max_depth: int = 8) -> list[Edge] | None:
        """Shortest directed path from ``start`` to ``target`` over ``predicates``."""
        wanted = set(predicates)
        queue: deque[tuple[str, list[Edge]]] = deque([(start, [])])
        seen = {start}
        while queue:
            node, trail = queue.popleft()
            if len(trail) >= max_depth:
                continue
            for edge in self.match(node, None, None):
                if edge.predicate not in wanted or edge.object in seen:
                    continue
                extended = trail + [edge]
                if edge.object == target:
                    return extended
                seen.add(edge.object)
                queue.append((edge.object, extended))
        return None

    def reachable(self, start: str, predicates: Iterable[str], max_depth: int = 8) -> set[str]:
        wanted = set(predicates)
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.match(node, None, None):
                if edge.predicate in wanted and edge.object not in seen:
                    seen.add(edge.object)
                    queue.append((edge.object, depth + 1))
        return seen

    # ---------------------------------------------------------------- context

    def context_for(self, focus: str | Iterable[str], hops: int = 2,
                    max_edges: int = 80) -> str:
        """Render a bounded neighbourhood as text.

        This is the agent's working memory: instead of carrying a transcript,
        it re-derives a compact view of the world from the graph on each turn.
        """
        roots = [focus] if isinstance(focus, str) else list(focus)
        frontier: list[str] = [r for r in roots]
        seen: set[str] = set(frontier)
        included: list[str] = list(frontier)
        for _ in range(hops):
            nxt: list[str] = []
            for node in frontier:
                for edge in list(self.match(node, None, None)) + self.in_edges(node):
                    for other in (edge.subject, edge.object):
                        if other not in seen:
                            seen.add(other)
                            included.append(other)
                            nxt.append(other)
            frontier = nxt

        lines: list[str] = []
        shown = 0
        for node_id in included:
            node = self._nodes.get(node_id)
            attrs = ""
            if node and node.attrs:
                attrs = " " + " ".join(f"{k}={v}" for k, v in sorted(node.attrs.items()))
            lines.append(f"{node_id} [{node.type if node else 'Unknown'}]{attrs}")
            for edge in self.match(node_id, None, None):
                if edge.object not in seen:
                    continue
                if shown >= max_edges:
                    lines.append("  ... (truncated)")
                    break
                marker = "" if edge.source == "observation" else f"  ({edge.source}"
                if marker:
                    marker += f" {edge.confidence:.2f})" if edge.confidence < 1.0 else ")"
                lines.append(f"  --{edge.predicate}--> {edge.object}{marker}")
                shown += 1
            if shown >= max_edges:
                break
        header = (f"# world model (focus: {', '.join(roots)}; {hops} hops; "
                  f"{len(included)} entities, rev {self.revision})")
        return "\n".join([header, *lines])

    # ------------------------------------------------------------ persistence

    def to_json(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "nodes": [n.to_json() for n in self._nodes.values()],
            "edges": [e.to_json() for e in self._edges.values()],
            "history": [e.to_json() for e in self._history],
            "journal": self.journal,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, raw: dict[str, Any], ontology: Any | None = None) -> KnowledgeGraph:
        kg = cls(ontology=ontology)
        for node in raw.get("nodes", []):
            kg._nodes[node["id"]] = Node.from_json(node)
        for edge_raw in raw.get("edges", []):
            edge = Edge.from_json(edge_raw)
            kg._edges[edge.triple] = edge
            if edge.live:
                kg._index(edge)
        kg._history = [Edge.from_json(e) for e in raw.get("history", [])]
        kg.journal = list(raw.get("journal", []))
        kg.revision = raw.get("revision", 0)
        return kg

    @classmethod
    def load(cls, path: str | Path, ontology: Any | None = None) -> KnowledgeGraph:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json(raw, ontology=ontology)

    # --------------------------------------------------------------- internal

    def _ensure_node(self, node_id: str, node_type: str | None) -> None:
        if node_id not in self._nodes:
            self.add_node(node_id, node_type or "Unknown")

    def _index(self, edge: Edge) -> None:
        s, p, o = edge.triple
        self._spo.setdefault(s, {}).setdefault(p, set()).add(o)
        self._pos.setdefault(p, {}).setdefault(s, set()).add(o)
        self._osp.setdefault(o, {}).setdefault(s, set()).add(p)

    def _unindex(self, edge: Edge) -> None:
        s, p, o = edge.triple
        for index, k1, k2, value in (
            (self._spo, s, p, o),
            (self._pos, p, s, o),
            (self._osp, o, s, p),
        ):
            bucket = index.get(k1, {}).get(k2)
            if bucket is not None:
                bucket.discard(value)
                if not bucket:
                    del index[k1][k2]
                    if not index[k1]:
                        del index[k1]

    def _log(self, op: str, **payload: Any) -> None:
        self.journal.append({"rev": self.revision, "op": op, **payload})

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        return (f"KnowledgeGraph(nodes={len(self._nodes)}, edges={len(self.edges)}, "
                f"rev={self.revision})")
