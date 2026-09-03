"""A research repository as the agent's world model.

Nothing here is typed in by hand either.  The graph is parsed out of a real
project -- ``ai4sci-molecule``, six weeks of molecular machine learning on ESOL
-- from its ``pyproject.toml``, its README, its ``.gitignore`` and the tables
committed under ``results/``.  The agent then plans the rebuild a fresh clone
needs, retries off the graph, and refuses what the model gets wrong about the
findings.  A bundled snapshot of the small result files is the default source,
so this runs anywhere; ``--repo PATH`` parses a real checkout instead.

    uv run python -m kg_agent.demo_study
    uv run python -m kg_agent.demo_study --repo /path/to/ai4sci-molecule
    uv run python -m kg_agent.demo_study --live   # OpenRouter; needs OPENROUTER_API_KEY
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
import tempfile
import tomllib
from pathlib import Path

from .agent import ActionResult, KGAgent
from .graph import KnowledgeGraph
from .llm import ScriptedLLM, claims
from .planner import Goal, plan_for
from .schema import Ontology, RelationSpec
from .verify import Claim, IngestPolicy, ingest, verify

#: An unmodified copy of the files below; see ``data/ai4sci_molecule/PROVENANCE.md``.
SNAPSHOT = Path(__file__).resolve().parent / "data" / "ai4sci_molecule"

#: node type -> the action that satisfies a node of that type.  ``Dataset`` is
#: deliberately absent: ``data/`` is gitignored, so the download is the blocked leaf.
STUDY_ACTIONS: dict[str, str] = {
    "Study": "run_study",
    "Split": "materialise_split",
    "Project": "reproduce",
}

RUN_COMMAND = re.compile(r"python -m (?P<package>[a-z0-9_]+)\.(?P<study>week\d+)")

#: Every study in this project sits on one dataset, so the graph names it once.
DATASET = "esol"


def study_ontology() -> Ontology:
    """Research-repository semantics.  Built from scratch; ~30 lines is a domain."""
    return Ontology(
        [
            RelationSpec("requires", transitive=True, dependency=True,
                         description="the subject cannot be reproduced without the object"),
            RelationSpec("evaluates", dependency=True, domain="Study", range="Split",
                         description="the study reports numbers on this split"),
            RelationSpec("drawn_from", dependency=True, functional=True, range="Dataset",
                         description="a split partitions exactly one dataset"),
            RelationSpec("derived_from", inverse="feeds", transitive=True, dependency=True,
                         domain="Study", range="Study",
                         incompatible_with=frozenset({"feeds"}),
                         description="the subject reads the object's outputs"),
            RelationSpec("feeds", inverse="derived_from", domain="Study", range="Study",
                         incompatible_with=frozenset({"derived_from"}),
                         description="materialised inverse; results never flow both ways"),
            RelationSpec("produced", inverse="produced_by", range="Artifact",
                         description="a file the study wrote and the repository committed"),
            RelationSpec("produced_by", inverse="produced", domain="Artifact"),
            RelationSpec("must_produce", range="Artifact",
                         description="gitignored output a fresh clone has to regenerate"),
            RelationSpec("outperforms", inverse="outperformed_by", transitive=True,
                         domain="Arm", range="Arm",
                         incompatible_with=frozenset({"outperformed_by"}),
                         description="lower error on the same table, same regime"),
            RelationSpec("outperformed_by", inverse="outperforms", domain="Arm", range="Arm",
                         incompatible_with=frozenset({"outperforms"})),
            RelationSpec("measures", domain="Arm",
                         description="the thing an arm of the sweep was measuring"),
            RelationSpec("uses_architecture", functional=True, range="Architecture",
                         description="the config pins one architecture per sweep"),
            RelationSpec("headline_strategy", functional=True, range="Strategy",
                         description="the strategy the study named before seeing numbers"),
            RelationSpec("limits", range="Caveat",
                         description="a stated limit on what the study can claim"),
            RelationSpec("satisfied_by",
                         description="a requirement already met by something concrete"),
        ],
        satisfied_states=("complete", "materialised", "rebuilt", "done"),
    )


# ---------------------------------------------------------------- the world


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], column: str) -> float | None:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError):
        return None


def _config_for(results: Path, study: str) -> dict:
    """The one ``*_config.json`` a week writes, unwrapped when it is nested."""
    for path in sorted((results / study).glob("*_config.json")):
        raw = _read_json(path)
        inner = raw.get("config")
        return inner if isinstance(inner, dict) else raw
    return {}


def _rank(kg: KnowledgeGraph, rows: list[dict[str, str]], *, column: str, name: str,
          kind: str, source: str, attrs: tuple[str, ...] = ()) -> None:
    """Turn one results table into regime-scoped arms ordered by error.

    Only rank-adjacent pairs are asserted; ``outperforms`` is transitive, so the
    rest of the order is derived rather than stored.  Arms are scoped by regime
    because the orderings genuinely disagree between them -- unscoped nodes
    would make the graph contradict itself.
    """
    by_regime: dict[str, list[tuple[float, str]]] = {}
    for row in rows:
        score = _number(row, column)
        if score is None:
            continue
        regime = row["regime"]
        arm = f"{regime}:{row[name]}"
        kg.add_node(arm, "Arm", regime=regime, kind=kind, **{column: round(score, 4)},
                    **{a: row[a] for a in attrs if row.get(a)})
        by_regime.setdefault(regime, []).append((score, arm))
    for regime, scored in by_regime.items():
        ordered = [arm for _, arm in sorted(scored)]
        for better, worse in zip(ordered, ordered[1:]):
            kg.assert_edge(better, "outperforms", worse, source=source)


def load_repo(path: str | Path = SNAPSHOT, ontology: Ontology | None = None) -> KnowledgeGraph:
    """Parse a research repository into a graph: studies, splits, artifacts, findings."""
    root = Path(path)
    kg = KnowledgeGraph(ontology=ontology or study_ontology())
    results = root / "results"
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    # 1. the project itself
    project_meta = _read_pyproject(root / "pyproject.toml")
    project = project_meta.get("name", root.name)
    kg.add_node(project, "Project", version=project_meta.get("version", ""),
                requires_python=project_meta.get("requires-python", ""))

    # 2. the studies, from the README's own reproduce commands
    studies = sorted({m["study"] for m in RUN_COMMAND.finditer(readme)})
    package = next((m["package"] for m in RUN_COMMAND.finditer(readme)), "")
    for study in studies:
        kg.add_node(study, "Study")
        kg.assert_edge(project, "requires", study, source="README.md")

    # 3. the dataset every split is drawn from, and the local copy that can satisfy it
    kg.add_node(DATASET, "Dataset", source="MoleculeNet")
    for study in sorted(results.glob("*/summary.json")) if results.exists() else []:
        stats = _read_json(study).get("scaffold_statistics")
        if stats:
            kg.update_node(DATASET, molecules=int(stats["num_molecules"]),
                           scaffolds=int(stats["num_scaffolds"]))
            break
    # `data/` is gitignored, so this path is the one concrete thing that can satisfy it.
    local = re.search(r"`(data/[\w/]+)/?`", readme)
    if local:
        kg.add_node(local.group(1).rstrip("/"), "LocalCopy", source="README.md")

    # 4. what each study evaluates, and what its config pins
    for study in studies:
        config = _config_for(results, study)
        seeds = config.get("seeds") or []
        if seeds:
            kg.update_node(study, seeds=f"{min(seeds)}-{max(seeds)}")
        # week4 trains nothing, so its config names split types through its regime map.
        split_types = (config.get("split_types")
                       or [pair[0] for pair in config.get("regime_map", [])])
        for split in split_types:
            kg.add_node(split, "Split")
            kg.assert_edge(study, "evaluates", split, source=f"{study} config")
            kg.assert_edge(split, "drawn_from", DATASET, source=f"{study} config")
        for architecture in config.get("architectures", []):
            kg.add_node(architecture, "Architecture")
        for strategy in config.get("strategies", []):
            kg.add_node(f"strategy:{strategy}", "Strategy")
        architecture = config.get("architecture") or config.get("headline_ensemble")
        if architecture:
            kg.add_node(architecture, "Architecture")
            kg.assert_edge(study, "uses_architecture", architecture, source=f"{study} config")
        headline = config.get("headline_strategy")
        if headline:
            kg.assert_edge(study, "headline_strategy", f"strategy:{headline}",
                           source=f"{study} config")

    # 5. splits are materialised when the checkout committed them
    for split in [n.id for n in kg.nodes if n.type == "Split"]:
        if any((results / s / "splits.json").exists() for s in studies):
            kg.update_node(split, status="materialised", source="splits.json")

    # 6. committed artifacts, and the gitignored ones a fresh clone lacks
    for study in studies:
        for entry in sorted((results / study).iterdir()) if (results / study).exists() else []:
            name = f"results/{study}/{entry.name}"
            if entry.is_dir():
                kg.add_node(name, "Artifact", status="committed", kind="shards",
                            files=len(list(entry.iterdir())))
            else:
                kg.add_node(name, "Artifact", status="committed", bytes=entry.stat().st_size)
            kg.assert_edge(study, "produced", name, source="results/ tree")
    for line in _gitignored(root):
        study = line.split("/")[1]
        if study in studies:
            kg.add_node(line, "Artifact", status="missing")
            kg.assert_edge(study, "must_produce", line, source=".gitignore")

    # 7. the findings: every ranked table a study committed, one arm per regime
    for study in studies:
        accuracy = results / study / "accuracy.csv"
        if accuracy.exists():
            _rank(kg, _read_csv(accuracy), column="ensemble_rmse_mean", name="ensemble",
                  kind="ensemble", source=f"results/{study}/accuracy.csv")
        budget = results / study / "budget_efficiency.csv"
        if not budget.exists():
            continue
        rows = _read_csv(budget)
        _rank(kg, rows, column="final_rmse", name="strategy", kind="strategy",
              source=f"results/{study}/budget_efficiency.csv",
              attrs=("speedup", "labels_saved", "reaches_target"))
        for row in rows:
            arm, strategy = f"{row['regime']}:{row['strategy']}", f"strategy:{row['strategy']}"
            if kg.has_node(arm) and kg.has_node(strategy):
                kg.assert_edge(arm, "measures", strategy,
                               source=f"results/{study}/budget_efficiency.csv")

    # 8. the question a study asked, and what it said it could not claim
    for study in studies:
        path = results / study / "summary.json"
        if not path.exists():
            continue
        summary = _read_json(path)
        if summary.get("question"):
            kg.update_node(study, question=summary["question"][:80])
        for i, caveat in enumerate(summary.get("caveats", []), start=1):
            kg.add_node(f"caveat:{study}-{i}", "Caveat", text=caveat[:60])
            kg.assert_edge(study, "limits", f"caveat:{study}-{i}",
                           source=f"results/{study}/summary.json")
        for row in summary.get("final_comparison", []):
            arm = f"{row['regime']}:{row['strategy']}"
            if kg.has_node(arm):
                kg.update_node(arm, excludes_zero=row["excludes_zero"], budget=row["budget"])

    # 9. a real checkout also states the study order in its imports
    for study in studies:
        module = root / "src" / package / f"{study}.py"
        if module.exists():
            for upstream in _imports(module):
                if upstream in studies and upstream != study:
                    kg.assert_edge(study, "derived_from", upstream, source=f"src/{package}")
        source = results / study / "summary.json"
        if source.exists():
            upstream = _read_json(source).get("source", {}).get("week3_directory")
            if upstream:
                kg.assert_edge(study, "derived_from", Path(upstream).name,
                               source=f"results/{study}/summary.json")

    # 10. a study is done when nothing it owns is missing
    for study in studies:
        if not any(kg.match(study, "must_produce", None)):
            kg.update_node(study, status="complete")
    return kg


def _read_pyproject(path: Path) -> dict:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})


def _gitignored(root: Path) -> list[str]:
    """The ``results/`` outputs the repository deliberately does not commit."""
    path = root / ".gitignore"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("results/")]


def _imports(module: Path) -> list[str]:
    """Sibling modules a week imports -- the study order, stated in code."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    return sorted({node.module for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module})


def project_of(kg: KnowledgeGraph) -> str:
    """The node the repository was parsed for."""
    for node in kg.nodes:
        if node.type == "Project":
            return node.id
    raise ValueError("no Project node in the graph -- is this a python project?")


def goal_for(kg: KnowledgeGraph) -> Goal:
    project = project_of(kg)
    return Goal(project, f"rebuild what a fresh clone of {project} is missing")


# ---------------------------------------------------------------- the agent


def scripted_llm() -> ScriptedLLM:
    """What a model would extract from each observation -- traps included."""
    return ScriptedLLM(
        # Asked which ready study goes first, the model front-loads the 1h37m
        # sweep; the planner alone would have gone alphabetically (week3 first).
        action=lambda plan, context: "run_study(week6)",
        claims_by_observation={
            "week3 sweep finished": claims("week3 produced results/week3/predictions.csv"),
            # Running week4 states a dependency the committed tables never do.
            "week4 error analysis": claims("week4 produced results/week4/molecule_errors.csv",
                                           "week4 derived_from week3"),
            "week5 ensembles": claims(
                "week5 produced results/week5/member_predictions.csv",
                # True in domain; the summary generalised it to the regime where
                # MLP is the better model.
                "out_of_scaffold:GIN outperforms out_of_scaffold:MLP",
            )
            # The notebook index has the study order backwards -- and by now the
            # graph has week4's dependency to refuse it with.
            + [Claim.parse("week3 derived_from week4", confidence=0.4)],
            "week6 resumed": claims(
                "week6 produced results/week6/test_predictions.csv",
                # The headline everyone expects, and the one the sweep refuted.
                "out_of_scaffold:uncertainty outperforms out_of_scaffold:random",
                # A strategy no config declares: the policy refuses to invent it.
                "out_of_scaffold:bald outperforms out_of_scaffold:random",
            ),
        },
        dependencies_by_node={
            "esol": claims(
                "esol satisfied_by data/MoleculeNet",
                # A dataset does not evaluate a split: domain violation.
                "esol evaluates random",
            ),
        },
    )


def build_agent(kg: KnowledgeGraph, llm=None) -> KGAgent:
    agent = KGAgent(kg, llm or scripted_llm(), actions=STUDY_ACTIONS, execute="stage",
                    policy=IngestPolicy(accept_unknown=True, unknown_confidence=0.6,
                                        allow_new_entities=False))

    observations = {
        "week3": "week3 sweep finished: week3 produced results/week3/predictions.csv. "
                 "The notebook index suggests week3 derived_from week4.",
        "week4": "week4 error analysis complete: week4 produced "
                 "results/week4/molecule_errors.csv, reading week3's predictions -- "
                 "week4 derived_from week3.",
        "week5": "week5 ensembles trained: week5 produced "
                 "results/week5/member_predictions.csv. GIN is the strongest ensemble, so "
                 "out_of_scaffold:GIN outperforms out_of_scaffold:MLP. The notebook index "
                 "lists week3 derived_from week4.",
        "week6": "week6 resumed from 31 committed trajectory shards and finished: week6 "
                 "produced results/week6/test_predictions.csv. Letting the model choose paid "
                 "off -- out_of_scaffold:uncertainty outperforms out_of_scaffold:random, and "
                 "out_of_scaffold:bald outperforms out_of_scaffold:random.",
    }

    @agent.action("run_study")
    def run_study(agent: KGAgent, step) -> ActionResult | str:
        node = agent.kg.get_node(step.node)
        if step.node == "week6":
            # Whether this is a retry is read off the world model, not off a
            # counter in this process -- so it survives the restart in section 4.
            if node is None or node.attrs.get("status") != "failed":
                return ActionResult(step, False,
                                    "week6 sweep interrupted: the session dropped after 31 "
                                    "of 50 trajectories", {"retry": True})
        for missing in agent.kg.match(step.node, "must_produce", None):
            agent.kg.update_node(missing.object, status="rebuilt")
        return observations.get(step.node, f"{step.node} rerun complete")

    @agent.action("materialise_split")
    def materialise_split(agent: KGAgent, step) -> str:
        users = sorted(e.subject for e in agent.kg.match(None, "evaluates", step.node))
        return (f"split materialised: {step.node} drawn_from esol, "
                f"evaluated by {', '.join(users)}")

    @agent.action("reproduce")
    def reproduce(agent: KGAgent, step) -> str:
        return f"uv sync && uv run python -m {step.node.replace('-', '_')}: every table rebuilt"

    return agent


# -------------------------------------------------------------- the sections


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def demo_world_model(kg: KnowledgeGraph, root: Path) -> None:
    banner("1. A research repository is the world model (parsed, not typed in)")
    print(f"source: {root}")
    print(f"{kg} -- every node and edge below came out of those files\n")
    print(kg.context_for("week6", hops=2))
    missing = sorted(e.object for e in kg.match(None, "must_produce", None))
    print(f"\ngitignored, so a fresh clone has to rebuild them: {', '.join(missing)}")
    print("This rendering -- rebuilt from the graph on every turn -- is what the model sees.")


def demo_planning(kg: KnowledgeGraph, llm=None) -> None:
    banner("2. The graph plans the rebuild, and vets what the model proposes")
    goal = goal_for(kg)
    plan = plan_for(kg, goal, actions=STUDY_ACTIONS)
    blocked = "esol" in plan.blocked
    print(plan.render())
    print("\nStages follow dependency depth: what a study evaluates comes before the study,")
    print("and every study comes before the project that requires it.")
    if blocked:
        print("No action satisfies a Dataset -- data/ is gitignored -- so esol is reported")
        print("blocked rather than guessed at, and the first turn has nothing it may run.")
    else:
        print("This checkout committed its splits.json, so the splits are already satisfied and")
        print("the subtree under them -- the dataset itself -- never enters the plan.")

    agent = build_agent(kg, llm)
    print(f"\nclaim extraction and step choice by: {type(agent.llm).__name__}"
          + (f" ({agent.llm.model})" if hasattr(agent.llm, "model") else ""))
    print("Each turn acts on the whole ready frontier (execute='stage'). Where several steps")
    print("are ready the model is asked which goes first; only a step on the plan is taken.")
    print("\n--- running the loop: observe -> update -> query -> plan -> choose -> act -> observe ---")
    run = agent.run(goal)
    print(run.render())
    week6 = [r for r in run.records if r.step and r.step.node == "week6"]
    if len(week6) > 1:
        print(f"\nturn {week6[0].turn} opened with week6 ({week6[0].chosen_by}'s choice), the sweep that")
        print("really took 1h37m; it was interrupted, so it fell into a later turn's frontier and")
        print("was retried off the graph's status=failed, not a counter in this process.")

    if blocked:
        banner("...how the blocked dataset got unblocked")
        print("Hitting the blocked leaf, the agent asked the model what esol needs.")
        print("Two proposals came back; the graph took one and refused the other:")
        edge = next(iter(kg.match("esol", "satisfied_by", None)), None)
        print(f"  accepted: {edge} (source={edge.source}, confidence={edge.confidence:.2f})"
              if edge else "  accepted: none")
        print(f"  refused : esol evaluates random -- in the graph? "
              f"{kg.has_edge('esol', 'evaluates', 'random')} "
              f"(esol is a Dataset; evaluates requires a Study)")

    order = next(iter(kg.match(None, "derived_from", None)), None)
    if order is None:
        return
    banner("...and what the run believes about the study order")
    edge = order
    print(f"one study reads another's outputs: {edge}")
    print(f"  source={edge.source}, confidence={edge.confidence:.2f}")
    if edge.source == "llm":
        print("  No committed table states this, so the model's claim was written provisionally")
        print("  when the run observed it -- and it held, because nothing in the graph disagreed.")
    else:
        print("  This checkout states it outright, so the run's claim came back SUPPORTED")
        print("  rather than being taken on trust.")
    print("\nEither way the reversed order was refused, and so was every claim that disagreed")
    print("with a parsed table. Each contradiction was kept, with the reason naming what it")
    print("collided with:")
    for resolution in run.resolutions:
        print(f"  {resolution.decision:<6} {resolution.verdict.claim}\n         {resolution.reason}")

    print("\nreplan after the run:")
    print(plan_for(kg, goal, actions=STUDY_ACTIONS).render())
    print(f"\nmessage history retained: 0 turns (the rebuild lives in the graph, rev {kg.revision})")


def demo_verification(kg: KnowledgeGraph) -> None:
    banner("3. The graph constrains what the agent believes about the findings")
    cases = [
        ("asserted by the sweep's own table",
         Claim.parse("out_of_scaffold:diversity outperforms out_of_scaffold:uncertainty_diversity")),
        ("derived: a transitive chain",
         Claim.parse("out_of_scaffold:diversity outperforms out_of_scaffold:random")),
        ("the inverse, materialised on assert",
         Claim.parse("out_of_scaffold:uncertainty outperformed_by out_of_scaffold:random")),
        ("the headline everyone expects",
         Claim.parse("out_of_scaffold:uncertainty outperforms out_of_scaffold:random")),
        ("a strategy no config declares",
         Claim.parse("out_of_scaffold:bald outperforms out_of_scaffold:random")),
        ("the config pins one architecture",
         Claim.parse("week6 uses_architecture GCN")),
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
    report = ingest(kg, [c for _, c in cases],
                    policy=IngestPolicy(accept_unknown=True, allow_new_entities=False))
    print(report.summary())
    for verdict in report.rejected:
        print(f"  refused: {verdict}")
    for edge in report.written:
        print(f"  written provisionally: {edge} (confidence {edge.confidence:.2f})")
    print(f"\nGIN is still what week6 pinned: {kg.has_edge('week6', 'uses_architecture', 'GIN')}; "
          f"bald never entered the graph: {not kg.has_node('out_of_scaffold:bald')}")

    print("\nWhat the ordering rests on, straight off the arms the tables produced:")
    arms = [n for n in kg.nodes if n.attrs.get("kind") == "strategy"
            and n.attrs.get("regime") == "out_of_scaffold"]
    for node in sorted(arms, key=lambda n: n.attrs["final_rmse"]):
        excludes = node.attrs.get("excludes_zero")
        print(f"  {node.id:<38} final_rmse={node.attrs['final_rmse']:<8} "
              f"speedup={node.attrs.get('speedup', '-'):<9} "
              f"ci excludes zero: {'baseline' if excludes is None else excludes}")
    print("No interval excludes zero, so the order is by point estimate -- the graph records")
    print("what the tables say, and the caveats the study attached to them:")
    for edge in kg.match("week6", "limits", None):
        print(f"  {kg.get_node(edge.object).attrs['text']}...")


def demo_restart(root: Path, llm=None) -> None:
    banner("4. The rebuild survives a restart (the graph is the whole handoff)")
    kg = load_repo(root)
    goal = goal_for(kg)
    first = build_agent(kg, llm)
    partial = first.run(goal, max_steps=6)
    acted = len([r for r in partial.records if r.step])
    print(f"process 1 stopped mid-stage after {acted} acted steps: {partial.reason}")
    print(partial.render())

    path = Path(tempfile.gettempdir()) / "kg_agent_study.json"
    kg.save(path)
    print(f"\nsaved to {path}")

    resumed = KnowledgeGraph.load(path, ontology=study_ontology())
    failed = sorted(n.id for n in resumed.nodes if n.attrs.get("status") == "failed")
    print(f"process 2 loaded a cold graph: {resumed} -- no transcript, no in-memory retry counters")
    print(f"carried over as failed: {', '.join(failed) or 'nothing'}")
    print(plan_for(resumed, goal, actions=STUDY_ACTIONS).render())

    second = build_agent(resumed, llm)
    finished = second.run(goal)
    print(f"\nprocess 2 finished the rebuild: completed={finished.completed} ({finished.reason})")
    print(finished.render())
    print("\nThe retry was not run blind: process 2 read week6's failed status off the graph")
    print("and resumed from the trajectory shards the repository commits on purpose.")


def main(argv: list[str] | None = None) -> None:
    """``--repo PATH`` picks the checkout; ``--live`` swaps in OpenRouter."""
    argv = sys.argv[1:] if argv is None else argv
    root = SNAPSHOT
    for i, arg in enumerate(argv):
        if arg == "--repo" and i + 1 < len(argv):
            root = Path(argv[i + 1])
        elif arg.startswith("--repo="):
            root = Path(arg.split("=", 1)[1])
    kg = load_repo(root)
    llm = None
    if "--live" in argv:
        from .openrouter import OpenRouterLLM
        model = next((a.split("=", 1)[1] for a in argv if a.startswith("--model=")), None)
        llm = OpenRouterLLM(model, ontology=kg.ontology)
    demo_world_model(kg, root)
    demo_planning(kg, llm)
    demo_verification(kg)
    demo_restart(root, llm)
    if llm is not None:
        print(f"\nOpenRouter usage: {llm.usage}")
    print()


if __name__ == "__main__":
    main()
