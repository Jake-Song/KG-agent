from kg_agent import (ActionResult, Goal, IngestPolicy, KGAgent, KnowledgeGraph, ScriptedLLM,
                      Status, claims, default_ontology)


def world() -> KnowledgeGraph:
    kg = KnowledgeGraph(ontology=default_ontology())
    kg.add_node("Hypothesis_A", "Hypothesis")
    kg.add_node("Measurement_B", "Measurement")
    kg.add_node("Instrument_C", "Instrument")
    kg.add_node("Lab_D", "Lab")
    kg.add_node("Protein_C", "Protein")
    kg.add_node("Protein_D", "Protein")
    kg.assert_edge("Hypothesis_A", "requires", "Measurement_B")
    kg.assert_edge("Measurement_B", "requires", "Instrument_C")
    kg.assert_edge("Instrument_C", "requires", "Lab_D")
    kg.assert_edge("Protein_C", "activates", "Protein_D")
    return kg


def build(kg: KnowledgeGraph, **kwargs) -> tuple[KGAgent, ScriptedLLM]:
    llm = ScriptedLLM(claims_by_observation={
        "measurement complete": claims(
            "Measurement_B produced Result_R12",   # unknown -> provisional
            "Protein_C inhibits Protein_D",        # hallucination -> refused
        ),
    })
    agent = KGAgent(kg, llm, **kwargs)
    for action in ("secure_lab_access", "acquire_instrument", "evaluate_hypothesis"):
        agent.register(action, lambda a, step: f"{step.action} done")
    agent.register("run_measurement", lambda a, step: "measurement complete")
    return agent, llm


def test_run_walks_the_dependency_chain_to_completion():
    kg = world()
    agent, _ = build(kg)
    run = agent.run(Goal("Hypothesis_A", "validate Hypothesis_A"))
    assert run.completed
    assert [r.step.node for r in run.records] == ["Lab_D", "Instrument_C", "Measurement_B",
                                                  "Hypothesis_A"]
    assert kg.get_node("Hypothesis_A").attrs["status"] == "complete"
    assert "completed" in run.render()


def test_action_results_land_in_the_graph_through_observe():
    kg = world()
    agent, _ = build(kg)
    agent.run("Hypothesis_A")
    edge = kg.get_edge("Measurement_B", "produced", "Result_R12")
    assert edge is not None and edge.source == "llm" and edge.confidence == 0.5


def test_a_hallucinated_claim_never_enters_the_world_model():
    kg = world()
    agent, _ = build(kg)
    run = agent.run("Hypothesis_A")
    assert not kg.has_edge("Protein_C", "inhibits", "Protein_D")
    assert kg.has_edge("Protein_C", "activates", "Protein_D")
    assert [v.claim.triple for v in run.contradictions] == [("Protein_C", "inhibits", "Protein_D")]
    assert run.contradictions[0].status is Status.CONTRADICTED


def test_a_strict_policy_can_refuse_unknown_claims_too():
    kg = world()
    agent, _ = build(kg, policy=IngestPolicy(accept_unknown=False))
    agent.run("Hypothesis_A")
    assert not kg.has_edge("Measurement_B", "produced", "Result_R12")


def test_the_model_sees_graph_context_never_a_transcript():
    kg = world()
    agent, llm = build(kg)
    agent.run("Hypothesis_A")
    contexts = [context for method, _, context in llm.calls if method == "propose_claims"]
    assert contexts and all(c.startswith("# world model") for c in contexts)
    # nothing on the agent accumulates turns; state is the graph alone
    assert not any(isinstance(v, list) and v and hasattr(v[0], "role")
                   for v in vars(agent).values())


def test_failed_actions_mark_the_node_and_are_retried_then_abandoned():
    kg = world()
    agent, _ = build(kg)
    agent.register("acquire_instrument", lambda a, step: False)
    run = agent.run("Hypothesis_A")
    assert not run.completed
    assert kg.get_node("Instrument_C").attrs["status"] == "failed"
    assert [r.step.node for r in run.records] == ["Lab_D", "Instrument_C", "Instrument_C"]
    assert "Instrument_C failed" in run.reason


def test_a_failure_that_recovers_replans_and_continues():
    kg = world()
    agent, _ = build(kg)
    attempts: list[str] = []

    def flaky(a, step):
        attempts.append(step.node)
        if len(attempts) == 1:
            return ActionResult(step, False, "instrument busy")
        return "instrument booked"

    agent.register("acquire_instrument", flaky)
    run = agent.run("Hypothesis_A")
    assert run.completed
    assert len(attempts) == 2


def test_a_blocked_leaf_asks_the_model_for_dependencies_and_verifies_them():
    kg = world()
    kg.add_node("Widget_X", "Widget")
    kg.assert_edge("Instrument_C", "requires", "Widget_X")
    llm = ScriptedLLM(
        claims_by_observation={},
        dependencies_by_node={"Widget_X": claims("Widget_X satisfied_by Spare_Part_9")},
    )
    agent = KGAgent(kg, llm)
    for action in ("secure_lab_access", "acquire_instrument", "run_measurement",
                   "evaluate_hypothesis"):
        agent.register(action, lambda a, step: f"{step.action} done")
    run = agent.run("Hypothesis_A")
    assert run.completed
    assert kg.has_edge("Widget_X", "satisfied_by", "Spare_Part_9")


def test_an_unhelpful_model_leaves_the_agent_honestly_stuck():
    kg = world()
    kg.add_node("Widget_X", "Widget")
    kg.assert_edge("Instrument_C", "requires", "Widget_X")
    agent, _ = build(kg)
    run = agent.run("Hypothesis_A")
    assert not run.completed
    assert run.reason == "no way to satisfy Widget_X"


def test_unregistered_action_is_reported_not_raised():
    kg = world()
    agent, _ = build(kg)
    agent._handlers.pop("secure_lab_access")
    run = agent.run("Hypothesis_A")
    assert not run.completed
    assert "no handler" in run.records[0].result.observation


def test_step_runs_exactly_one_cycle():
    kg = world()
    agent, _ = build(kg)
    record = agent.step("Hypothesis_A", iteration=1)
    assert record.step.node == "Lab_D"
    assert record.result.ok
    assert kg.get_node("Lab_D").attrs["status"] == "complete"
    assert [s.node for s in agent.plan("Hypothesis_A").steps] == ["Instrument_C", "Measurement_B",
                                                                 "Hypothesis_A"]
