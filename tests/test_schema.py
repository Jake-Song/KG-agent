import pytest

from kg_agent import ConstraintViolation, KnowledgeGraph, Ontology, RelationSpec, default_ontology


def kg() -> KnowledgeGraph:
    return KnowledgeGraph(ontology=default_ontology())


def test_functional_relation_rebinds_and_retracts_the_old_object():
    graph = kg()
    graph.assert_edge("Instrument_C", "located_at", "Lab_D")
    result = graph.assert_edge("Instrument_C", "located_at", "Lab_E")
    assert result.status == "replaced"
    assert [e.triple for e in result.retracted] == [("Instrument_C", "located_at", "Lab_D")]
    assert graph.has_edge("Instrument_C", "located_at", "Lab_E")
    assert not graph.has_edge("Instrument_C", "located_at", "Lab_D")


def test_mutually_exclusive_predicate_is_refused():
    graph = kg()
    graph.assert_edge("Protein_A", "activates", "Protein_B")
    result = graph.assert_edge("Protein_A", "inhibits", "Protein_B")
    assert result.status == "rejected"
    assert not result
    assert [e.triple for e in result.conflicts] == [("Protein_A", "activates", "Protein_B")]
    assert not graph.has_edge("Protein_A", "inhibits", "Protein_B")


def test_strict_mode_raises():
    graph = kg()
    graph.assert_edge("Protein_A", "activates", "Protein_B")
    with pytest.raises(ConstraintViolation):
        graph.assert_edge("Protein_A", "inhibits", "Protein_B", strict=True)


def test_inverse_and_symmetric_edges_are_materialised():
    graph = kg()
    result = graph.assert_edge("Protein_A", "inhibits", "Protein_B")
    assert graph.has_edge("Protein_B", "inhibited_by", "Protein_A")
    assert [e.source for e in result.entailed] == ["inference"]

    graph.assert_edge("H", "conflicts_with", "P")
    assert graph.has_edge("P", "conflicts_with", "H")


def test_domain_and_range_violations_are_reported():
    graph = kg()
    graph.add_node("Dataset_A", "Dataset")
    graph.add_node("Protein_B", "Protein")
    errors = graph.ontology.type_errors(graph, "Dataset_A", "inhibits", "Protein_B")
    assert errors and "requires Protein" in errors[0]
    assert graph.assert_edge("Dataset_A", "inhibits", "Protein_B").status == "rejected"


def test_unknown_typed_nodes_do_not_trip_type_checks():
    graph = kg()
    graph.add_node("X")  # type Unknown
    assert graph.assert_edge("X", "inhibits", "Protein_B")


def test_incompatibility_is_made_symmetric_both_ways():
    onto = Ontology([RelationSpec("a", incompatible_with=frozenset({"b"})), RelationSpec("b")])
    assert onto.incompatible("b") == {"a"}
    assert onto.incompatible("a") == {"b"}


def test_inverses_are_registered_in_both_directions():
    onto = Ontology([RelationSpec("above", inverse="below")])
    assert onto.inverse_of("above") == "below"
    assert onto.inverse_of("below") == "above"


def test_dependency_predicates_drive_the_planner():
    onto = default_ontology()
    assert "requires" in onto.dependency_predicates
    assert "located_at" not in onto.dependency_predicates
