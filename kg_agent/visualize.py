"""Render a :class:`KnowledgeGraph` as Mermaid or Graphviz DOT text.

Both formats are text based, so the utility has no runtime dependencies and can
be pasted into Markdown, Mermaid Live, or rendered later with Graphviz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from .graph import KnowledgeGraph
from .research_domain import ResearchOntology

__all__ = ["to_mermaid", "to_dot", "render", "load_graph"]


def _quote(value: object) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _node_label(node) -> str:
    attrs = " ".join(f"{key}={value}" for key, value in sorted(node.attrs.items()))
    return f"{node.id} [{node.type}{(' ' + attrs) if attrs else ''}]"


def _mermaid_ids(node_ids):
    """Map graph IDs to unique Mermaid-safe identifiers."""
    mapping = {}
    used = set()
    for node_id in sorted(node_ids):
        safe = re.sub(r"[^A-Za-z0-9_]", "_", node_id)
        if not safe or safe[0].isdigit():
            safe = "node_" + safe
        candidate = safe
        suffix = 2
        while candidate in used:
            candidate = f"{safe}_{suffix}"
            suffix += 1
        mapping[node_id] = candidate
        used.add(candidate)
    return mapping


def _selected(kg, focus=None, hops=None):
    if focus is None:
        nodes = {node.id for node in kg.nodes}
    else:
        roots = [focus] if isinstance(focus, str) else list(focus)
        if hops is None:
            nodes = set(roots)
            for edge in kg.edges:
                nodes.update((edge.subject, edge.object))
        else:
            nodes = set(roots)
            frontier = set(roots)
            for _ in range(max(0, hops)):
                next_frontier = set()
                for edge in kg.edges:
                    if edge.subject in frontier or edge.object in frontier:
                        next_frontier.update((edge.subject, edge.object))
                next_frontier -= nodes
                nodes.update(next_frontier)
                frontier = next_frontier
    return nodes


def _edges(kg, nodes, include_retracted=False):
    edges = []
    for edge in kg.edges:
        if edge.subject in nodes and edge.object in nodes and (include_retracted or edge.live):
            edges.append(edge)
    if include_retracted:
        edges.extend(edge for edge in kg.history if not edge.live and edge.subject in nodes and edge.object in nodes)
    return sorted({(e.subject, e.predicate, e.object, e.source, e.confidence, e.retracted_at): e for e in edges}.values(),
                  key=lambda e: (e.subject, e.predicate, e.object, e.retracted_at is not None, e.source))


def to_mermaid(kg, *, focus=None, hops=None, include_retracted=False):
    """Return a deterministic Mermaid flowchart for live graph edges."""
    nodes = _selected(kg, focus, hops)
    mermaid_ids = _mermaid_ids(nodes)
    lines = ["flowchart LR"]
    for node_id in sorted(nodes):
        node = kg.get_node(node_id)
        if node is not None:
            lines.append(f"  {mermaid_ids[node_id]}[{_quote(_node_label(node))}]")
    for edge in _edges(kg, nodes, include_retracted):
        marker = " (retracted)" if not edge.live else ""
        lines.append(f"  {mermaid_ids[edge.subject]} -->|{_quote(edge.predicate + marker)}| {mermaid_ids[edge.object]}")
    return "\n".join(lines) + "\n"


def to_dot(kg, *, focus=None, hops=None, include_retracted=False):
    """Return a deterministic Graphviz DOT directed graph."""
    nodes = _selected(kg, focus, hops)
    lines = ["digraph KnowledgeGraph {", "  rankdir=LR;", "  node [shape=box];"]
    for node_id in sorted(nodes):
        node = kg.get_node(node_id)
        if node is not None:
            lines.append(f"  {_quote(node_id)} [label={_quote(_node_label(node))}];")
    for edge in _edges(kg, nodes, include_retracted):
        label = edge.predicate + (" (retracted)" if not edge.live else "")
        style = ", style=dashed, color=gray" if not edge.live else ""
        lines.append(f"  {_quote(edge.subject)} -> {_quote(edge.object)} [label={_quote(label)}{style}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render(kg, format="mermaid", **kwargs):
    if format == "mermaid":
        return to_mermaid(kg, **kwargs)
    if format == "dot":
        return to_dot(kg, **kwargs)
    raise ValueError(f"unsupported format: {format!r}")


def load_graph(path):
    """Load a bare graph JSON or a research session's ``state.json``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "graph" in raw:
        return KnowledgeGraph.from_json(raw["graph"], ontology=ResearchOntology())
    return KnowledgeGraph.from_json(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="KnowledgeGraph JSON or research state.json")
    parser.add_argument("--format", choices=("mermaid", "dot"), default="mermaid")
    parser.add_argument("--output", type=Path, help="Write the drawing here; stdout otherwise")
    parser.add_argument("--focus", action="append", help="Node to include (repeatable)")
    parser.add_argument("--hops", type=int, help="Hop limit around --focus")
    parser.add_argument("--include-retracted", action="store_true")
    args = parser.parse_args(argv)
    try:
        text = render(load_graph(args.input), args.format, focus=args.focus, hops=args.hops,
                      include_retracted=args.include_retracted)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
