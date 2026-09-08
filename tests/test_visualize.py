import json

from kg_agent import KnowledgeGraph
from kg_agent.visualize import load_graph, render, to_dot, to_mermaid


def graph():
    kg = KnowledgeGraph()
    kg.add_node("A", "Thing", status="ready")
    kg.add_node("B", "Thing")
    kg.assert_edge("A", "requires", "B", source="observation")
    kg.retract_edge("A", "requires", "B")
    return kg


def test_mermaid_is_deterministic_and_escapes_labels():
    kg = graph()
    kg.add_node('C"quoted', "Thing")
    text = to_mermaid(kg)
    assert text.startswith("flowchart LR\n")
    assert "C_quoted" in text
    assert "retracted" not in text
    assert to_mermaid(kg) == text
    assert "A" in text and "B" in text


def test_dot_includes_retracted_edges_as_dashed_when_requested():
    text = to_dot(graph(), include_retracted=True)
    assert '"A" -> "B"' in text
    assert "dashed" in text and "requires (retracted)" in text
    assert to_dot(graph(), include_retracted=False).count("->") == 0


def test_focus_and_hops_limit_graph():
    kg = KnowledgeGraph()
    kg.assert_edge("A", "p", "B")
    kg.assert_edge("B", "p", "C")
    kg.assert_edge("C", "p", "D")
    text = to_mermaid(kg, focus="A", hops=1)
    assert "A[" in text and "B[" in text
    assert "C[" not in text and "D[" not in text


def test_render_and_load_bare_and_research_graphs(tmp_path):
    kg = graph()
    bare = tmp_path / "kg.json"
    kg.save(bare)
    assert load_graph(bare).has_node("A")
    session = tmp_path / "state.json"
    session.write_text(json.dumps({"graph": kg.to_json()}))
    assert load_graph(session).has_node("A")
    assert render(kg, "dot").startswith("digraph KnowledgeGraph")
