from kg_agent import KnowledgeGraph, Ontology, RelationSpec, default_ontology


def simple_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node("a", "Thing")
    kg.assert_edge("a", "p", "b")
    kg.assert_edge("a", "q", "c")
    kg.assert_edge("d", "p", "b")
    return kg


def test_match_uses_every_pattern_shape():
    kg = simple_kg()
    assert {e.triple for e in kg.match("a", None, None)} == {("a", "p", "b"), ("a", "q", "c")}
    assert {e.triple for e in kg.match(None, "p", None)} == {("a", "p", "b"), ("d", "p", "b")}
    assert {e.triple for e in kg.match(None, None, "b")} == {("a", "p", "b"), ("d", "p", "b")}
    assert [e.triple for e in kg.match("a", "p", "b")] == [("a", "p", "b")]
    assert [e.triple for e in kg.match("a", "p", "zzz")] == []
    assert len(list(kg.match())) == 3


def test_object_creates_node_with_ontology_type():
    kg = KnowledgeGraph(ontology=default_ontology())
    kg.assert_edge("Exp", "uses_dataset", "Dataset_A")
    assert kg.node_type("Dataset_A") == "Dataset"


def test_retraction_is_soft():
    kg = simple_kg()
    retired = kg.retract_edge("a", "p", "b")
    assert retired is not None and not retired.live
    assert not kg.has_edge("a", "p", "b")
    assert [e.triple for e in kg.match(None, None, "b")] == [("d", "p", "b")]
    assert ("a", "p", "b") in {e.triple for e in kg.history}
    assert kg.retract_edge("a", "p", "b") is None


def test_reasserting_a_retracted_edge_revives_it():
    kg = simple_kg()
    kg.retract_edge("a", "p", "b")
    assert kg.assert_edge("a", "p", "b").status == "added"
    assert kg.has_edge("a", "p", "b")


def test_journal_records_every_mutation_in_order():
    kg = simple_kg()
    kg.update_node("a", status="failed")
    kg.retract_edge("a", "p", "b")
    ops = [entry["op"] for entry in kg.journal]
    assert ops[0] == "add_node"
    assert "assert_edge" in ops
    assert ops[-1] == "retract_edge"
    assert [e["rev"] for e in kg.journal] == sorted(e["rev"] for e in kg.journal)


def test_json_round_trip_preserves_state(tmp_path):
    kg = simple_kg()
    kg.update_node("a", status="failed")
    kg.retract_edge("a", "q", "c")
    path = kg.save(tmp_path / "kg.json")

    reloaded = KnowledgeGraph.load(path)
    assert sorted(e.triple for e in reloaded.edges) == sorted(e.triple for e in kg.edges)
    assert reloaded.get_node("a").attrs == {"status": "failed"}
    assert reloaded.revision == kg.revision
    assert reloaded.journal == kg.journal
    assert not reloaded.has_edge("a", "q", "c")
    # indexes were rebuilt, not just the dict
    assert [e.triple for e in reloaded.match(None, None, "b")] == [("a", "p", "b"),
                                                                  ("d", "p", "b")]


def test_context_for_is_hop_bounded():
    kg = KnowledgeGraph()
    kg.assert_edge("a", "p", "b")
    kg.assert_edge("b", "p", "c")
    kg.assert_edge("c", "p", "d")
    near = kg.context_for("a", hops=1)
    assert "b" in near and "--p--> c" not in near
    far = kg.context_for("a", hops=3)
    assert "--p--> d" in far


def test_context_for_shows_attributes_and_provenance():
    kg = KnowledgeGraph()
    kg.add_node("a", "Thing", status="failed")
    kg.assert_edge("a", "p", "b", source="llm", confidence=0.4)
    text = kg.context_for("a", hops=1)
    assert "a [Thing] status=failed" in text
    assert "(llm 0.40)" in text


def test_path_and_reachable_follow_only_named_predicates():
    kg = KnowledgeGraph()
    kg.assert_edge("a", "requires", "b")
    kg.assert_edge("b", "requires", "c")
    kg.assert_edge("c", "other", "d")
    assert [e.triple for e in kg.path("a", "c", ["requires"])] == [("a", "requires", "b"),
                                                                  ("b", "requires", "c")]
    assert kg.path("a", "d", ["requires"]) is None
    assert kg.reachable("a", ["requires"]) == {"b", "c"}


def test_confidence_is_promoted_not_downgraded():
    kg = KnowledgeGraph()
    kg.assert_edge("a", "p", "b", confidence=0.4, source="llm")
    assert kg.assert_edge("a", "p", "b", confidence=0.9, source="observation").status == "updated"
    edge = kg.get_edge("a", "p", "b")
    assert edge.confidence == 0.9 and edge.source == "observation"
    assert kg.assert_edge("a", "p", "b", confidence=0.2, source="llm").status == "updated"
    assert kg.get_edge("a", "p", "b").confidence == 0.9


def test_empty_ontology_permits_any_predicate():
    kg = KnowledgeGraph(ontology=Ontology())
    assert kg.assert_edge("a", "anything_at_all", "b")
    kg2 = KnowledgeGraph(ontology=Ontology([RelationSpec("p")]))
    assert not kg2.assert_edge("a", "q", "b")
