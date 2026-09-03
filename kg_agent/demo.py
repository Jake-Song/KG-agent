"""Runnable walkthrough of the three capabilities over one shared world model.

    uv run python -m kg_agent.demo
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .agent import KGAgent, ActionResult
from .graph import KnowledgeGraph
from .llm import ScriptedLLM, claims
from .planner import Goal, plan_for
from .schema import default_ontology
from .verify import Claim, IngestPolicy, ingest, verify


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def build_world() -> KnowledgeGraph:
    """The environment the agent lives in -- state, not a document store."""
    kg = KnowledgeGraph(ontology=default_ontology())

    kg.add_node("Experiment_42", "Experiment", status="failed")
    kg.add_node("Dataset_A", "Dataset")
    kg.add_node("Hypothesis_H3", "Hypothesis")
    kg.add_node("Result_R7", "Result")
    kg.add_node("Paper_P9", "Paper")
    kg.add_node("Assumption_A2", "Assumption")
    kg.assert_edge("Experiment_42", "uses_dataset", "Dataset_A", strict=True)
    kg.assert_edge("Experiment_42", "tests", "Hypothesis_H3", strict=True)
    kg.assert_edge("Experiment_42", "produced", "Result_R7", strict=True)
    kg.assert_edge("Hypothesis_H3", "conflicts_with", "Paper_P9", strict=True)
    kg.assert_edge("Hypothesis_H3", "depends_on", "Assumption_A2", strict=True)

    # The dependency chain the planner will walk.  Dataset_B sits beside the
    # lab at stage 0, so the frontier has two ready steps for the model to order.
    kg.add_node("Hypothesis_A", "Hypothesis")
    kg.add_node("Measurement_B", "Measurement")
    kg.add_node("Instrument_C", "Instrument")
    kg.add_node("Lab_D", "Lab")
    kg.add_node("Dataset_B", "Dataset")
    kg.assert_edge("Hypothesis_A", "requires", "Measurement_B", strict=True)
    kg.assert_edge("Hypothesis_A", "requires", "Dataset_B", strict=True)
    kg.assert_edge("Measurement_B", "requires", "Instrument_C", strict=True)
    kg.assert_edge("Instrument_C", "located_at", "Lab_D", strict=True)
    kg.assert_edge("Instrument_C", "requires", "Lab_D", strict=True)  # access is a prerequisite

    # Molecular facts for the claim-checking section.
    for protein in ("Protein_A", "Protein_B", "Protein_C", "Protein_D", "Protein_E", "Protein_F",
                    "Protein_G", "Protein_H"):
        kg.add_node(protein, "Protein")
    kg.assert_edge("Protein_A", "inhibits", "Protein_B", strict=True)
    kg.assert_edge("Protein_C", "activates", "Protein_D", strict=True)
    return kg


def demo_world_model(kg: KnowledgeGraph) -> None:
    banner("1. The KG as world model (persistent state, not a transcript)")
    print(kg.context_for("Experiment_42", hops=2))
    print("\nThis rendering -- not a message history -- is what the agent hands the model.")

    path = Path(tempfile.gettempdir()) / "kg_agent_world.json"
    kg.save(path)
    reloaded = KnowledgeGraph.load(path, ontology=default_ontology())
    print(f"\nsaved to {path} and reloaded: {reloaded}")
    print(f"  edges identical: {sorted(e.triple for e in reloaded.edges) == sorted(e.triple for e in kg.edges)}")
    print(f"  journal entries replayable: {len(reloaded.journal)}")
    print(f"  symmetric edge materialised on assert: "
          f"{reloaded.has_edge('Paper_P9', 'conflicts_with', 'Hypothesis_H3')}")


def build_agent(kg: KnowledgeGraph, llm=None) -> KGAgent:
    llm = llm or ScriptedLLM(
        claims_by_observation={
            "lab access granted": claims("Lab_D satisfied_by Access_Badge_17")
            # A low-confidence aside from the lab notes: written provisionally.
            + [Claim.parse("Protein_G activates Protein_H", confidence=0.4)],
            "instrument booked": claims("Instrument_C located_at Lab_D"),
            "measurement complete": claims(
                "Measurement_B produced Result_R12",
                "Result_R12 supports Hypothesis_A",
                # A hallucination: the graph already says Protein_C activates Protein_D.
                "Protein_C inhibits Protein_D",
            )
            # Contradicts the provisional aside above, with much higher confidence.
            + [Claim.parse("Protein_G inhibits Protein_H", confidence=0.9)],
            "hypothesis evaluated": claims("Result_R12 supports Hypothesis_A"),
        },
        dependencies_by_node={},
        # Asked which ready step goes first, the model front-loads lab access (the
        # longest lead time); the planner alone would have loaded Dataset_B first.
        action=lambda plan, context: "secure_lab_access(Lab_D)",
    )
    agent = KGAgent(kg, llm, execute="stage",
                    policy=IngestPolicy(accept_unknown=True, unknown_confidence=0.6))
    attempts = {"Instrument_C": 0}

    @agent.action("secure_lab_access")
    def secure_lab_access(agent: KGAgent, step) -> str:
        return ("lab access granted: Lab_D satisfied_by Access_Badge_17. "
                "A note on the bench suggests Protein_G activates Protein_H.")

    @agent.action("load_dataset")
    def load_dataset(agent: KGAgent, step) -> str:
        return f"dataset loaded: {step.node}"

    @agent.action("acquire_instrument")
    def acquire_instrument(agent: KGAgent, step) -> ActionResult:
        attempts[step.node] = attempts.get(step.node, 0) + 1
        if attempts[step.node] == 1:
            return ActionResult(step, False, f"{step.node} is in use by another team",
                                {"retry": True})
        return ActionResult(step, True,
                            "instrument booked: Instrument_C located_at Lab_D")

    @agent.action("run_measurement")
    def run_measurement(agent: KGAgent, step) -> str:
        return ("measurement complete: Measurement_B produced Result_R12, "
                "and Result_R12 supports Hypothesis_A. "
                "Separately, Protein_C inhibits Protein_D, and the assay shows "
                "Protein_G inhibits Protein_H.")

    @agent.action("evaluate_hypothesis")
    def evaluate_hypothesis(agent: KGAgent, step) -> str:
        agent.kg.update_node(step.node, status="validated")
        return "hypothesis evaluated: Result_R12 supports Hypothesis_A"

    return agent


def demo_planning(kg: KnowledgeGraph, llm=None) -> None:
    banner("2. The KG as planner state (backward chaining over dependencies)")
    goal = Goal("Hypothesis_A", "validate Hypothesis_A")
    print(plan_for(kg, goal).render())

    agent = build_agent(kg, llm)
    print(f"\nclaim extraction and step choice by: {type(agent.llm).__name__}"
          + (f" ({agent.llm.model})" if hasattr(agent.llm, "model") else ""))
    print("Each turn acts on the whole ready frontier (execute='stage'); with two steps ready")
    print("the model is asked which goes first, and only a step on the plan is accepted.")
    print("\n--- running the loop: observe -> update -> query -> plan -> choose -> act -> observe ---")
    run = agent.run(goal)
    print(run.render())

    print("\ncontradictions were resolved against provenance, not dropped:")
    for resolution in run.resolutions:
        print(f"  {resolution.decision:<8} {resolution.verdict.claim} -- {resolution.reason}")
    print(f"  Protein_G activates Protein_H live: {kg.has_edge('Protein_G', 'activates', 'Protein_H')}; "
          f"inhibits: {kg.has_edge('Protein_G', 'inhibits', 'Protein_H')}; "
          f"retracted edges in history: {len([e for e in kg.history if e.retracted_at])}")
    print(f"  Protein_C activates Protein_D still live: {kg.has_edge('Protein_C', 'activates', 'Protein_D')}")

    print("\nreplan after the run:")
    print(plan_for(kg, goal).render())
    print(f"\nagent message history retained: 0 turns "
          f"(state lives in the graph, now at rev {kg.revision})")


def demo_verification(kg: KnowledgeGraph) -> None:
    banner("3. The KG as a constraint on hallucination")
    cases = [
        ("edge exists", Claim.parse("Protein_A inhibits Protein_B")),
        ("nothing known", Claim.parse("Protein_E inhibits Protein_F")),
        ("graph says the opposite", Claim.parse("Protein_C inhibits Protein_D")),
        ("type violation", Claim.parse("Dataset_A inhibits Protein_B")),
        ("derivable, never asserted", Claim.parse("Hypothesis_A requires Instrument_C")),
    ]
    for label, claim in cases:
        verdict = verify(kg, claim)
        print(f"\n{label}:")
        print(f"  claim   : {claim}")
        print(f"  status  : {verdict.status.upper()}")
        print(f"  because : {verdict.explanation}")
        for edge in verdict.evidence:
            print(f"  evidence: {edge}")

    banner("...and the gate that keeps the world model clean")
    report = ingest(kg, [c for _, c in cases], policy=IngestPolicy(accept_unknown=True))
    print(report.summary())
    for verdict in report.rejected:
        print(f"  refused: {verdict}")
    for edge in report.written:
        print(f"  written provisionally: {edge} (confidence {edge.confidence:.2f})")
    print(f"\nProtein_C still activates Protein_D: "
          f"{kg.has_edge('Protein_C', 'activates', 'Protein_D')}; "
          f"the contradicted claim never entered the graph: "
          f"{not kg.has_edge('Protein_C', 'inhibits', 'Protein_D')}")


def main(argv: list[str] | None = None) -> None:
    """``--live`` swaps the scripted model for OpenRouter (needs OPENROUTER_API_KEY)."""
    argv = sys.argv[1:] if argv is None else argv
    kg = build_world()
    llm = None
    if "--live" in argv:
        from .openrouter import OpenRouterLLM
        model = next((a.split("=", 1)[1] for a in argv if a.startswith("--model=")), None)
        llm = OpenRouterLLM(model, ontology=kg.ontology)
    demo_world_model(kg)
    demo_planning(kg, llm)
    demo_verification(kg)
    if llm is not None:
        print(f"\nOpenRouter usage: {llm.usage}")
    print()


if __name__ == "__main__":
    main()
