import pytest

from kg_agent import (CyclicDependencyError, Goal, KnowledgeGraph, default_ontology, plan_for)


def chain() -> KnowledgeGraph:
    kg = KnowledgeGraph(ontology=default_ontology())
    kg.add_node("Hypothesis_A", "Hypothesis")
    kg.add_node("Measurement_B", "Measurement")
    kg.add_node("Instrument_C", "Instrument")
    kg.add_node("Lab_D", "Lab")
    kg.assert_edge("Hypothesis_A", "requires", "Measurement_B")
    kg.assert_edge("Measurement_B", "requires", "Instrument_C")
    kg.assert_edge("Instrument_C", "requires", "Lab_D")
    kg.assert_edge("Instrument_C", "located_at", "Lab_D")
    return kg


def test_dependencies_come_before_dependents():
    plan = plan_for(chain(), Goal("Hypothesis_A", "validate Hypothesis_A"))
    assert [s.node for s in plan.steps] == ["Lab_D", "Instrument_C", "Measurement_B",
                                            "Hypothesis_A"]
    assert [s.action for s in plan.steps] == ["secure_lab_access", "acquire_instrument",
                                              "run_measurement", "evaluate_hypothesis"]
    assert plan.next_step().node == "Lab_D"
    assert not plan.complete


def test_non_dependency_predicates_are_not_traversed():
    kg = chain()
    kg.retract_edge("Instrument_C", "requires", "Lab_D")
    plan = plan_for(kg, "Hypothesis_A")
    assert "Lab_D" not in [s.node for s in plan.steps]  # located_at alone is not a dependency


def test_stages_group_independent_work():
    kg = chain()
    kg.add_node("Calibration_E", "Measurement")
    kg.assert_edge("Measurement_B", "requires", "Calibration_E")
    plan = plan_for(kg, "Hypothesis_A")
    stage_names = [sorted(s.node for s in stage) for stage in plan.stages]
    assert ["Calibration_E", "Lab_D"] == stage_names[0]
    assert stage_names[-1] == ["Hypothesis_A"]
    assert [s.stage for s in plan.steps] == sorted(s.stage for s in plan.steps)


def test_satisfied_subtrees_are_pruned():
    kg = chain()
    kg.update_node("Instrument_C", status="available")
    plan = plan_for(kg, "Hypothesis_A")
    assert [s.node for s in plan.steps] == ["Measurement_B", "Hypothesis_A"]
    assert plan.satisfied == ["Instrument_C"]
    assert "Lab_D" not in [s.node for s in plan.steps]


def test_satisfied_by_edge_also_prunes():
    kg = chain()
    kg.assert_edge("Lab_D", "satisfied_by", "Access_Badge_17")
    plan = plan_for(kg, "Hypothesis_A")
    assert "Lab_D" not in [s.node for s in plan.steps]
    assert plan.satisfied == ["Lab_D"]


def test_satisfied_goal_yields_an_empty_plan():
    kg = chain()
    kg.update_node("Hypothesis_A", status="validated")
    plan = plan_for(kg, "Hypothesis_A")
    assert plan.complete and plan.steps == []
    assert plan.next_step() is None


def test_cycles_are_reported_with_the_cycle():
    kg = KnowledgeGraph(ontology=default_ontology())
    kg.assert_edge("A", "requires", "B")
    kg.assert_edge("B", "requires", "C")
    kg.assert_edge("C", "requires", "A")
    with pytest.raises(CyclicDependencyError) as excinfo:
        plan_for(kg, "A")
    assert excinfo.value.cycle[0] == excinfo.value.cycle[-1] == "A"


def test_leaves_with_no_known_action_are_blocked():
    kg = KnowledgeGraph(ontology=default_ontology())
    kg.add_node("Hypothesis_A", "Hypothesis")
    kg.add_node("Widget_X", "Widget")
    kg.assert_edge("Hypothesis_A", "requires", "Widget_X")
    plan = plan_for(kg, "Hypothesis_A")
    assert plan.blocked == ["Widget_X"]
    assert plan.next_step().blocked
    assert plan.next_step().action == "resolve"
    assert "blocked on: Widget_X" in plan.render()


def test_custom_action_map_unblocks():
    kg = KnowledgeGraph(ontology=default_ontology())
    kg.add_node("Hypothesis_A", "Hypothesis")
    kg.add_node("Widget_X", "Widget")
    kg.assert_edge("Hypothesis_A", "requires", "Widget_X")
    plan = plan_for(kg, "Hypothesis_A", actions={"Widget": "order_widget",
                                                 "Hypothesis": "evaluate_hypothesis"})
    assert plan.blocked == []
    assert plan.steps[0].action == "order_widget"


def test_render_is_readable():
    text = plan_for(chain(), Goal("Hypothesis_A", "validate Hypothesis_A")).render()
    assert "goal: validate Hypothesis_A" in text
    assert "stage 0: secure_lab_access(Lab_D)" in text
