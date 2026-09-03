import pytest

from kg_agent import (ActionResult, Claim, Goal, IngestPolicy, KGAgent, KnowledgeGraph,
                      ScriptedLLM, Status, claims, default_ontology)


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
    # The graph's edge came from an observation, so the agent kept it.
    assert [r.decision for r in run.resolutions] == ["kept"]
    assert run.unresolved == run.contradictions
    assert "kept:" in run.render() and "resolved:" not in run.render()


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


# ------------------------------------------------------------ model choice


def branching(action=None) -> tuple[KnowledgeGraph, KGAgent, ScriptedLLM]:
    """world() plus a second stage-0 leaf, so the frontier has two ready steps."""
    kg = world()
    kg.add_node("Dataset_A", "Dataset")
    kg.assert_edge("Hypothesis_A", "requires", "Dataset_A")
    llm = ScriptedLLM(action=action)
    agent = KGAgent(kg, llm)
    for name in ("load_dataset", "secure_lab_access", "acquire_instrument", "run_measurement",
                 "evaluate_hypothesis"):
        agent.register(name, lambda a, step: f"{step.action} done")
    return kg, agent, llm


def test_the_model_picks_among_ready_steps():
    kg, agent, llm = branching(action=lambda plan, ctx: "secure_lab_access")
    run = agent.run("Hypothesis_A")
    assert run.completed
    assert run.records[0].step.node == "Lab_D" and run.records[0].chosen_by == "model"
    assert run.records[1].step.node == "Dataset_A" and run.records[1].chosen_by == "planner"
    # Asked once per turn with a frontier of two: (Dataset_A, Lab_D) then (Dataset_A, Instrument_C).
    contexts = [ctx for method, _, ctx in llm.calls if method == "choose_action"]
    assert len(contexts) == 2 and all(c.startswith("# world model") for c in contexts)
    assert "(model's choice)" in run.render()


def test_an_off_plan_choice_falls_back_to_the_planner():
    kg, agent, llm = branching(action=lambda plan, ctx: "launch_rocket")
    run = agent.run("Hypothesis_A")
    assert run.completed
    assert run.records[0].step.node == "Dataset_A" and run.records[0].chosen_by == "planner"


def test_choose_action_is_not_consulted_for_a_single_ready_step():
    kg = world()
    agent, llm = build(kg)
    agent.run("Hypothesis_A")
    assert not any(method == "choose_action" for method, _, _ in llm.calls)


# ------------------------------------------------------------- stage mode


def test_stage_mode_acts_on_the_whole_frontier_per_turn():
    kg, agent, _ = branching()
    agent.execute = "stage"
    run = agent.run("Hypothesis_A")
    assert run.completed
    assert [(r.turn, r.step.node) for r in run.records] == [
        (1, "Dataset_A"), (1, "Lab_D"), (2, "Instrument_C"), (3, "Measurement_B"),
        (4, "Hypothesis_A")]
    assert [r.iteration for r in run.records] == [1, 2, 3, 4, 5]
    assert "turn 1: frontier of 2" in run.render()


def test_step_mode_render_has_no_turn_headers():
    kg = world()
    agent, _ = build(kg)
    assert "turn " not in agent.run("Hypothesis_A").render()


def test_stage_mode_stops_mid_stage_when_the_budget_runs_out():
    kg, agent, _ = branching()
    agent.execute = "stage"
    run = agent.run("Hypothesis_A", max_steps=1)
    assert not run.completed and "budget" in run.reason
    assert [r.step.node for r in run.records] == ["Dataset_A"]
    assert kg.get_node("Lab_D").attrs.get("status") is None


def test_stage_mode_skips_blocked_steps_then_unblocks():
    kg = world()
    kg.add_node("Widget_X", "Widget")
    kg.assert_edge("Instrument_C", "requires", "Widget_X")
    llm = ScriptedLLM(dependencies_by_node={"Widget_X": claims("Widget_X satisfied_by Spare_Part_9")})
    agent = KGAgent(kg, llm, execute="stage")
    for name in ("secure_lab_access", "acquire_instrument", "run_measurement",
                 "evaluate_hypothesis"):
        agent.register(name, lambda a, step: f"{step.action} done")
    run = agent.run("Hypothesis_A")
    assert run.completed
    assert [(r.turn, r.step.node) for r in run.records if r.step] == [
        (1, "Lab_D"), (3, "Instrument_C"), (4, "Measurement_B"), (5, "Hypothesis_A")]
    unblock = [r for r in run.records if r.step is None]
    assert len(unblock) == 1 and unblock[0].turn == 2 and "blocked on Widget_X" in unblock[0].note
    assert "turn 2: blocked on Widget_X; asked the model, which added 1 edge" in run.render()


def test_stage_mode_retries_a_failed_step_on_the_next_turn_then_abandons():
    kg, agent, _ = branching()
    agent.execute = "stage"
    agent.register("acquire_instrument", lambda a, step: False)
    run = agent.run("Hypothesis_A")
    assert not run.completed
    assert [r.step.node for r in run.records] == ["Dataset_A", "Lab_D", "Instrument_C",
                                                  "Instrument_C"]
    assert [r.turn for r in run.records] == [1, 1, 2, 3]
    assert run.reason == "Instrument_C failed 2 times"


def test_a_run_that_finishes_on_its_last_budgeted_step_is_completed():
    kg = world()
    agent, _ = build(kg)
    run = agent.run("Hypothesis_A", max_steps=4)
    assert run.completed and run.reason == "goal satisfied"


def test_invalid_execute_mode_raises():
    with pytest.raises(ValueError):
        KGAgent(world(), ScriptedLLM(), execute="parallel")


# ------------------------------------------------- contradiction resolution


def provisional_world(confidence: float) -> KnowledgeGraph:
    kg = world()
    kg.retract_edge("Protein_C", "activates", "Protein_D")
    kg.retract_edge("Protein_D", "activated_by", "Protein_C")
    kg.assert_edge("Protein_C", "activates", "Protein_D", source="llm", confidence=confidence)
    return kg


def resolving_agent(kg: KnowledgeGraph, claim_confidence: float, **kwargs) -> KGAgent:
    llm = ScriptedLLM(claims_by_observation={
        "measurement complete": [Claim.parse("Protein_C inhibits Protein_D",
                                             confidence=claim_confidence)],
    })
    agent = KGAgent(kg, llm, **kwargs)
    for name in ("secure_lab_access", "acquire_instrument", "evaluate_hypothesis"):
        agent.register(name, lambda a, step: f"{step.action} done")
    agent.register("run_measurement", lambda a, step: "measurement complete")
    return agent


def test_a_higher_confidence_observation_replaces_a_provisional_model_edge():
    kg = provisional_world(0.4)
    run = resolving_agent(kg, 0.9).run("Hypothesis_A")
    resolution = run.resolutions[0]
    assert resolution.decision == "replaced"
    assert kg.has_edge("Protein_C", "inhibits", "Protein_D")
    assert kg.has_edge("Protein_D", "inhibited_by", "Protein_C")
    assert not kg.has_edge("Protein_C", "activates", "Protein_D")
    assert not kg.has_edge("Protein_D", "activated_by", "Protein_C")      # the twin cascaded
    assert sorted(e.predicate for e in resolution.retracted) == ["activated_by", "activates"]
    assert kg.get_edge("Protein_C", "inhibits", "Protein_D").confidence == 0.9
    assert run.unresolved == [] and len(run.contradictions) == 1
    assert any(entry["op"] == "retract_edge" and entry["reason"] == "superseded by observation"
               for entry in kg.journal)
    assert "resolved: replaced" in run.render()


def test_a_lower_confidence_claim_does_not_replace_a_provisional_edge():
    kg = provisional_world(0.8)
    run = resolving_agent(kg, 0.5).run("Hypothesis_A")
    assert run.resolutions[0].decision == "kept"
    assert "at least as confident" in run.resolutions[0].reason
    assert kg.has_edge("Protein_C", "activates", "Protein_D")
    assert not kg.has_edge("Protein_C", "inhibits", "Protein_D")


def test_a_strict_policy_never_replaces():
    kg = provisional_world(0.4)
    run = resolving_agent(kg, 0.9, policy=IngestPolicy(accept_unknown=False)).run("Hypothesis_A")
    assert run.resolutions[0].decision == "kept"
    assert "policy" in run.resolutions[0].reason
    assert kg.has_edge("Protein_C", "activates", "Protein_D")


def test_a_custom_resolver_overrides_the_default():
    kg = world()                                       # observation-sourced, confidence 1.0
    agent = resolving_agent(kg, 0.9)

    @agent.resolver
    def trust_the_latest(agent, verdict):
        return "replace" if verdict.claim.predicate == "inhibits" else None

    run = agent.run("Hypothesis_A")
    assert run.resolutions[0].decision == "replaced"
    assert run.resolutions[0].reason == "decided by the resolver hook"
    assert kg.has_edge("Protein_C", "inhibits", "Protein_D")
    assert not kg.has_edge("Protein_C", "activates", "Protein_D")

    kg = world()
    agent = resolving_agent(kg, 0.9)
    agent.on_contradiction(lambda agent, verdict: None)      # defer to the default rule
    run = agent.run("Hypothesis_A")
    assert run.resolutions[0].decision == "kept"
    assert kg.has_edge("Protein_C", "activates", "Protein_D")
