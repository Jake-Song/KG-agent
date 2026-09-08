"""Replanning: the plan is recomputed from the graph, never patched.

    uv run python -m kg_agent.demo_replan
    uv run python -m kg_agent.demo_replan --live   # OpenRouter; needs OPENROUTER_API_KEY

An agent that stores its plan has to decide what to do when the world stops
matching it.  This one stores no plan at all.  Every turn calls :func:`plan_for`
again on whatever the graph now says, so a discovered dependency *adds* a step,
a discovered ``satisfied_by`` *deletes* a whole subtree, and a retraction brings
that subtree back -- with no plan-repair code anywhere.  The only thing that
moves a plan is an edge that survived the ingest gate.
"""

from __future__ import annotations

import sys

from .agent import ActionResult, KGAgent
from .graph import KnowledgeGraph
from .llm import ScriptedLLM, claims
from .planner import CyclicDependencyError, Goal, Plan, plan_for
from .schema import default_ontology
from .verify import IngestPolicy

GOAL = Goal("Hypothesis_H", "validate Hypothesis_H")

#: What each action reports.  The handlers return these and the model reads them
#: back, so the observation the agent acts on and the one it learns from cannot drift.
LAB = ("lab access granted: badge issued for Lab_L. The archive index says the assay "
       "was already run last month -- Measurement_M satisfied_by Result_Archive.")
NO_DATA = ("dataset unavailable: nothing has produced it yet -- Dataset_D requires "
           "Experiment_E, the calibration run.")
CALIBRATED = ("calibration run finished and Dataset_D is staged. The log footer also "
              "claims Hypothesis_H requires Instrument_Z.")
LOADED = "dataset staged: Dataset_D is on disk."
EVALUATED = "hypothesis evaluated: Result_Archive supports Hypothesis_H."


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def build_world() -> KnowledgeGraph:
    """Lab access comes first; two entities have no dependencies yet."""
    kg = KnowledgeGraph(ontology=default_ontology())
    kg.add_node("Hypothesis_H", "Hypothesis")
    kg.add_node("Measurement_M", "Measurement")
    kg.add_node("Instrument_I", "Instrument")
    kg.add_node("Lab_L", "Lab")
    kg.add_node("Dataset_D", "Dataset")
    kg.assert_edge("Hypothesis_H", "requires", "Measurement_M", strict=True)
    kg.assert_edge("Hypothesis_H", "requires", "Dataset_D", strict=True)
    kg.assert_edge("Measurement_M", "requires", "Instrument_I", strict=True)
    kg.assert_edge("Instrument_I", "requires", "Lab_L", strict=True)
    kg.assert_edge("Dataset_D", "requires", "Lab_L", strict=True)
    # Known to the lab, wired to nothing.  The run discovers what they are for --
    # and because they are already entities, the strict policy below can still
    # refuse the ones the model invents.
    kg.add_node("Experiment_E", "Experiment")
    kg.add_node("Result_Archive", "Result")
    return kg


def scripted_llm() -> ScriptedLLM:
    """What a model would extract from each observation."""
    return ScriptedLLM(
        claims_by_observation={
            # Prunes three steps: what is satisfied is not planned for.
            LAB: claims("Measurement_M satisfied_by Result_Archive"),
            # Adds one the first plan never contained.
            NO_DATA: claims("Dataset_D requires Experiment_E"),
            # A dependency on an instrument no one has heard of: refused.
            CALIBRATED: claims("Hypothesis_H requires Instrument_Z"),
            LOADED: [],
            # Accepted, and still no effect: `supports` is not a dependency.
            EVALUATED: claims("Result_Archive supports Hypothesis_H"),
        },
    )


def build_agent(kg: KnowledgeGraph, llm=None) -> KGAgent:
    agent = KGAgent(kg, llm or scripted_llm(), execute="stage",
                    policy=IngestPolicy(accept_unknown=True, unknown_confidence=0.6,
                                        allow_new_entities=False))

    @agent.action("secure_lab_access")
    def secure_lab_access(agent: KGAgent, step) -> str:
        return LAB

    @agent.action("load_dataset")
    def load_dataset(agent: KGAgent, step) -> ActionResult | str:
        # Whether this is the retry is read off the world model, not off a
        # counter in this process -- the graph is the only state there is.
        node = agent.kg.get_node(step.node)
        if node is None or node.attrs.get("status") != "failed":
            return ActionResult(step, False, NO_DATA, {"retry": True})
        return LOADED

    @agent.action("run_experiment")
    def run_experiment(agent: KGAgent, step) -> str:
        return CALIBRATED

    @agent.action("evaluate_hypothesis")
    def evaluate_hypothesis(agent: KGAgent, step) -> str:
        agent.kg.update_node(step.node, status="validated")
        return EVALUATED

    # Never reached once the archive prunes them on turn 1; registered so the
    # plan the agent is handed is one it could actually have carried out.
    @agent.action("acquire_instrument")
    def acquire_instrument(agent: KGAgent, step) -> str:
        return f"instrument booked: {step.node}"

    @agent.action("run_measurement")
    def run_measurement(agent: KGAgent, step) -> str:
        return f"measurement complete: {step.node}"

    return agent


# ------------------------------------------------------------------ plan diffs


def _why_gone(node: str, after: Plan, kg: KnowledgeGraph) -> str:
    if node in after.satisfied:
        return "satisfied, so the planner stopped there"
    entity = kg.get_node(node)
    if entity is not None and entity.attrs.get("status") in kg.ontology.satisfied_states:
        return "done"
    return "unreachable: the only path to it was pruned"


def plan_diff(before: Plan, after: Plan, kg: KnowledgeGraph) -> list[str]:
    """What changed between two plans -- printed for the reader, never applied.

    The agent never computes this: it replaces the plan wholesale.
    """
    was = {s.node: s for s in before.steps}
    now = {s.node: s for s in after.steps}
    lines = [f"  + {now[n]}  (new step, stage {now[n].stage})" for n in sorted(set(now) - set(was))]
    lines += [f"  - {was[n]}  ({_why_gone(n, after, kg)})" for n in sorted(set(was) - set(now))]
    lines += [f"  ~ {now[n]}  (stage {was[n].stage} -> {now[n].stage})"
              for n in sorted(set(was) & set(now)) if was[n].stage != now[n].stage]
    return lines


# ------------------------------------------------------------------- sections


def demo_initial_plan(kg: KnowledgeGraph) -> None:
    banner("1. The plan the graph implies right now")
    print(plan_for(kg, GOAL).render())
    print("\nBackward-chained from the goal over `requires`; stages follow dependency depth.")
    print("Nothing about this plan is stored -- it is a view of the graph, like a query result.")
    print("Experiment_E and Result_Archive are in the graph too, and nothing depends on them,")
    print("so no stage mentions them.")


def demo_run(kg: KnowledgeGraph, llm=None):
    banner("2. Replanning through a run: one plan per turn, built from scratch")
    agent = build_agent(kg, llm)
    run = agent.run(GOAL)

    plans: dict[int, Plan] = {}
    for record in run.records:
        plans.setdefault(record.turn, record.plan)

    previous: Plan | None = None
    for turn, plan in sorted(plans.items()):
        print(f"\n--- turn {turn} {'-' * 52}")
        print(plan.render())
        if previous is not None:
            print("changed since the previous turn:")
            print("\n".join(plan_diff(previous, plan, kg) or ["  (nothing)"]))
        for record in run.records:
            if record.turn != turn or record.step is None or record.result is None:
                continue
            mark = "ok" if record.result.ok else "failed"
            chosen = " (model's choice)" if record.chosen_by == "model" else ""
            print(f"  act {record.step} -> {mark}{chosen}")
            print(f"      observed: {record.result.observation}")
            if record.report:
                print(f"      ingest  : {record.report.summary()}")
        previous = plan

    print(f"\n--- after the last turn {'-' * 42}")
    print(run.plan.render() if run.plan else "")
    print(f"outcome: {'completed' if run.completed else 'stopped'} ({run.reason})")
    acted = len([r for r in run.records if r.step])
    print(f"\n{len(plans)} plans built and thrown away over {acted} acted steps. The two main")
    print("changes went in opposite directions: turn 2 lost a subtree the archive made")
    print("unnecessary; turn 3 gained a step the first plan never contained. No plan-repair")
    print(f"code ran, because no plan survived a turn (graph now at rev {kg.revision}).")
    return run


def demo_non_moves(kg: KnowledgeGraph, run) -> None:
    banner("3. Three ways a claim fails to move the plan")
    rejected = [v for r in run.records if r.report for v in r.report.rejected]
    print("a) refused at the ingest gate -- it never reached the graph:")
    for verdict in rejected:
        print(f"   {verdict}")
    print(f"   Instrument_Z in the graph: {kg.has_node('Instrument_Z')}")
    print("   A dependency on an entity nothing declares would have added a stage; the")
    print("   policy (allow_new_entities=False) never gave the planner the chance.")

    print("\nb) written, and still inert -- the planner only walks dependency edges:")
    edge = kg.get_edge("Result_Archive", "supports", "Hypothesis_H")
    print(f"   {edge} (source={edge.source}, confidence={edge.confidence:.2f})")
    print(f"   traversed by the planner: {', '.join(sorted(kg.ontology.dependency_predicates))}")
    print("   `supports` is not one of them, so the graph grew and the plan did not.")

    print("\nc) accepted, a dependency, and there is still no plan to be had:")
    cyclic = build_world()
    cyclic.assert_edge("Dataset_D", "requires", "Hypothesis_H", source="llm", confidence=0.6)
    print("   wrote Dataset_D requires Hypothesis_H -- no ontology rule forbids it")
    try:
        plan_for(cyclic, GOAL)
    except CyclicDependencyError as exc:
        print(f"   plan_for: {type(exc).__name__}: {exc}")
    print("   The planner is the second gate, and it refuses rather than looping.")
    cyclic.retract_edge("Dataset_D", "requires", "Hypothesis_H", reason="cyclic")
    print(f"   retracted; the plan comes straight back: "
          f"{len(plan_for(cyclic, GOAL).stages)} stages")


def demo_retraction(llm=None) -> None:
    banner("4. The plan follows the graph in both directions")
    kg = build_world()
    build_agent(kg, llm).run(GOAL, max_steps=1)   # turn 1 only
    before = plan_for(kg, GOAL)
    print("after turn 1, with the archive edge believed:")
    print(before.render())

    gone = kg.retract_edge("Measurement_M", "satisfied_by", "Result_Archive",
                           reason="the archive entry was a different assay")
    print(f"\nretracted {gone} -- a provisional model claim (source={gone.source}, "
          f"confidence {gone.confidence:.2f}) that three steps rested on")
    after = plan_for(kg, GOAL)
    print(after.render())
    print("changed:")
    print("\n".join(plan_diff(before, after, kg)))
    print("\nThe subtree came back on its own: nothing had to remember that it was once")
    print("planned. Lab_L stays out -- it is done, not pruned -- so the work already")
    print("finished is not repeated, and the retracted edge is still in the history.")
    print(f"retracted edges kept: {len([e for e in kg.history if e.retracted_at])}")


def main(argv: list[str] | None = None) -> None:
    """``--live`` swaps the scripted model for OpenRouter (needs OPENROUTER_API_KEY)."""
    argv = sys.argv[1:] if argv is None else argv
    kg = build_world()
    llm = None
    if "--live" in argv:
        from .openrouter import OpenRouterLLM
        model = next((a.split("=", 1)[1] for a in argv if a.startswith("--model=")), None)
        llm = OpenRouterLLM(model, ontology=kg.ontology)
    demo_initial_plan(kg)
    run = demo_run(kg, llm)
    demo_non_moves(kg, run)
    demo_retraction(llm)
    if llm is not None:
        print(f"\nOpenRouter usage: {llm.usage}")
    print()


if __name__ == "__main__":
    main()
