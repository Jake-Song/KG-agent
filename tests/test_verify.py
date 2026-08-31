from kg_agent import (Claim, IngestPolicy, KnowledgeGraph, Status, default_ontology, ingest,
                      verify, verify_all)


def kg() -> KnowledgeGraph:
    graph = KnowledgeGraph(ontology=default_ontology())
    for protein in ("Protein_A", "Protein_B", "Protein_C", "Protein_D", "Protein_E", "Protein_F"):
        graph.add_node(protein, "Protein")
    graph.add_node("Dataset_A", "Dataset")
    graph.assert_edge("Protein_A", "inhibits", "Protein_B")
    graph.assert_edge("Protein_C", "activates", "Protein_D")
    return graph


def test_claim_parse():
    assert Claim.parse("Protein_A inhibits Protein_B").triple == ("Protein_A", "inhibits",
                                                                 "Protein_B")
    assert Claim.parse("a --p--> b").triple == ("a", "p", "b")


def test_supported():
    verdict = verify(kg(), Claim.parse("Protein_A inhibits Protein_B"))
    assert verdict.status is Status.SUPPORTED
    assert verdict.trusted
    assert [e.triple for e in verdict.evidence] == [("Protein_A", "inhibits", "Protein_B")]


def test_unknown():
    verdict = verify(kg(), Claim.parse("Protein_E inhibits Protein_F"))
    assert verdict.status is Status.UNKNOWN
    assert not verdict.trusted
    assert verdict.evidence == []


def test_contradicted_carries_the_conflicting_edge_as_evidence():
    verdict = verify(kg(), Claim.parse("Protein_C inhibits Protein_D"))
    assert verdict.status is Status.CONTRADICTED
    assert [e.triple for e in verdict.evidence] == [("Protein_C", "activates", "Protein_D")]
    assert "activates" in verdict.explanation


def test_ill_formed_on_type_violation():
    verdict = verify(kg(), Claim.parse("Dataset_A inhibits Protein_B"))
    assert verdict.status is Status.ILL_FORMED


def test_ill_formed_on_unknown_predicate_and_entity():
    graph = kg()
    assert verify(graph, Claim.parse("Protein_A frobnicates Protein_B")).status is Status.ILL_FORMED
    assert verify(graph, Claim.parse("Protein_A inhibits Nowhere")).status is Status.ILL_FORMED
    lenient = verify(graph, Claim.parse("Protein_A inhibits Nowhere"),
                     require_known_entities=False)
    assert lenient.status is Status.UNKNOWN


def test_entailed_by_transitivity():
    graph = KnowledgeGraph(ontology=default_ontology())
    graph.assert_edge("A", "requires", "B")
    graph.assert_edge("B", "requires", "C")
    verdict = verify(graph, Claim.parse("A requires C"))
    assert verdict.status is Status.ENTAILED
    assert [e.triple for e in verdict.derivation] == [("A", "requires", "B"),
                                                      ("B", "requires", "C")]


def test_entailed_by_symmetry():
    graph = KnowledgeGraph(ontology=default_ontology())
    graph.assert_edge("H", "conflicts_with", "P")
    # drop the asserted direction; only the materialised mirror remains
    graph.retract_edge("H", "conflicts_with", "P")
    verdict = verify(graph, Claim.parse("H conflicts_with P"))
    assert verdict.status is Status.ENTAILED
    assert [e.triple for e in verdict.evidence] == [("P", "conflicts_with", "H")]


def test_functional_rebinding_reads_as_contradiction_for_a_claim():
    graph = KnowledgeGraph(ontology=default_ontology())
    graph.assert_edge("Instrument_C", "located_at", "Lab_D")
    graph.add_node("Lab_E", "Lab")
    verdict = verify(graph, Claim.parse("Instrument_C located_at Lab_E"))
    assert verdict.status is Status.CONTRADICTED


def test_verify_all_preserves_order():
    graph = kg()
    verdicts = verify_all(graph, [Claim.parse("Protein_A inhibits Protein_B"),
                                  Claim.parse("Protein_E inhibits Protein_F")])
    assert [v.status for v in verdicts] == [Status.SUPPORTED, Status.UNKNOWN]


def test_policy_writes_unknown_provisionally():
    graph = kg()
    report = ingest(graph, [Claim.parse("Protein_E inhibits Protein_F")],
                    policy=IngestPolicy(unknown_confidence=0.3))
    assert len(report.provisional) == 1
    edge = graph.get_edge("Protein_E", "inhibits", "Protein_F")
    assert edge.confidence == 0.3 and edge.source == "llm"
    assert graph.has_edge("Protein_F", "inhibited_by", "Protein_E")


def test_policy_can_refuse_unknown():
    graph = kg()
    report = ingest(graph, [Claim.parse("Protein_E inhibits Protein_F")],
                    policy=IngestPolicy(accept_unknown=False))
    assert len(report.rejected) == 1
    assert not graph.has_edge("Protein_E", "inhibits", "Protein_F")


def test_policy_never_writes_a_contradiction():
    graph = kg()
    report = ingest(graph, [Claim.parse("Protein_C inhibits Protein_D")])
    assert len(report.contradictions) == 1
    assert not graph.has_edge("Protein_C", "inhibits", "Protein_D")
    assert graph.has_edge("Protein_C", "activates", "Protein_D")
    assert report.written == []


def test_policy_reports_known_claims_without_rewriting():
    graph = kg()
    before = graph.revision
    report = ingest(graph, [Claim.parse("Protein_A inhibits Protein_B")])
    assert len(report.accepted) == 1
    assert graph.revision == before


def test_new_entities_are_gated_by_the_policy():
    graph = kg()
    strict = ingest(graph, [Claim.parse("Protein_A inhibits Newcomer")],
                    policy=IngestPolicy(allow_new_entities=False))
    assert strict.rejected[0].status is Status.ILL_FORMED
    assert not graph.has_node("Newcomer")

    lenient = ingest(graph, [Claim.parse("Protein_A inhibits Newcomer")],
                     policy=IngestPolicy(allow_new_entities=True))
    assert len(lenient.provisional) == 1
    assert graph.node_type("Newcomer") == "Protein"


def test_report_summary_counts():
    graph = kg()
    report = ingest(graph, [Claim.parse("Protein_A inhibits Protein_B"),
                            Claim.parse("Protein_E inhibits Protein_F"),
                            Claim.parse("Protein_C inhibits Protein_D")])
    assert report.summary() == "1 known, 1 provisional, 1 rejected (1 contradicted)"
    assert len(report.verdicts) == 3
