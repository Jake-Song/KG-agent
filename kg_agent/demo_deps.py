"""A practical walkthrough: a real ``uv.lock`` as the agent's world model.

Nothing here is typed in by hand.  The graph is parsed from a lockfile --
this repo's own by default, or any uv project's via ``--lock PATH`` -- and
the agent plans an upgrade over the real dependency tree, retries off the
graph, refuses what the model gets wrong about the tree, and survives a
restart.  It is also the worked example for an :class:`~kg_agent.schema.Ontology`
that lives outside the package: ``default_ontology()`` is not involved.

    uv run python -m kg_agent.demo_deps
    uv run python -m kg_agent.demo_deps --lock /path/to/other/uv.lock
    uv run python -m kg_agent.demo_deps --live      # OpenRouter; needs OPENROUTER_API_KEY
"""

from __future__ import annotations

import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import urlparse

from .agent import ActionResult, KGAgent
from .graph import KnowledgeGraph
from .llm import ScriptedLLM, claims
from .planner import Goal, plan_for
from .schema import Ontology, RelationSpec
from .verify import Claim, IngestPolicy, ingest, verify

REPO_LOCK = Path(__file__).resolve().parents[1] / "uv.lock"

#: node type -> the action that satisfies a node of that type.  ``Runtime``
#: is deliberately absent: the interpreter is the plan's blocked leaf.
DEPS_ACTIONS: dict[str, str] = {
    "Package": "upgrade_package",
    "Project": "resync_project",
}


def lock_ontology() -> Ontology:
    """Dependency-tree semantics.  Built from scratch; ~30 lines is a domain."""
    return Ontology(
        [
            RelationSpec("depends_on", inverse="required_by", transitive=True, dependency=True,
                         range="Package", incompatible_with=frozenset({"required_by"}),
                         description="the subject cannot be installed without the object"),
            RelationSpec("required_by", inverse="depends_on", domain="Package",
                         incompatible_with=frozenset({"depends_on"}),
                         description="materialised inverse; a dependency never runs both ways"),
            RelationSpec("pinned_to", functional=True, range="Version",
                         description="the lockfile binds each package to exactly one version"),
            RelationSpec("runs_on", dependency=True, range="Runtime",
                         description="the interpreter the project needs"),
            RelationSpec("published_at", functional=True, range="Registry",
                         description="the index a package was resolved from"),
            RelationSpec("conflicts_with", symmetric=True,
                         description="two packages that cannot be installed together"),
            RelationSpec("satisfied_by",
                         description="a requirement already met by something concrete"),
        ],
        satisfied_states=("complete", "available", "current", "upgraded", "done"),
    )


# ---------------------------------------------------------------- the world


def load_lock(path: str | Path = REPO_LOCK, ontology: Ontology | None = None) -> KnowledgeGraph:
    """Parse a ``uv.lock`` into a graph: packages, pins, registries, the tree.

    A sibling ``.python-version`` file, if present, becomes an ``Interpreter``
    node -- the one concrete thing that can satisfy the ``Runtime`` requirement.
    """
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    kg = KnowledgeGraph(ontology=ontology or lock_ontology())
    packages = raw.get("package", [])

    # Nodes first, so every dependency edge lands on a typed node.
    for pkg in packages:
        name, version, source = pkg["name"], pkg.get("version", ""), pkg.get("source", {})
        if "virtual" in source or "editable" in source:
            kg.add_node(name, "Project", version=version)
        else:
            kg.add_node(name, "Package", version=version, status="locked")

    python = raw.get("requires-python")
    if python:
        kg.add_node("python", "Runtime", requires=python)
        pinned = path.with_name(".python-version")
        if pinned.exists() and pinned.read_text().strip():
            kg.add_node(f"cpython-{pinned.read_text().strip()}", "Interpreter",
                        source=pinned.name)

    for pkg in packages:
        name, version, source = pkg["name"], pkg.get("version", ""), pkg.get("source", {})
        if version:
            kg.add_node(f"{name}=={version}", "Version")
            kg.assert_edge(name, "pinned_to", f"{name}=={version}", source="uv.lock")
        if "registry" in source:
            host = urlparse(source["registry"]).netloc or source["registry"]
            kg.add_node(host, "Registry")
            kg.assert_edge(name, "published_at", host, source="uv.lock")
        if python and kg.node_type(name) == "Project":
            kg.assert_edge(name, "runs_on", "python", source="uv.lock")

        groups = [("uv.lock", pkg.get("dependencies", []))]
        for group, deps in pkg.get("dev-dependencies", {}).items():
            groups.append((f"uv.lock:{group}", deps))
        for source_tag, deps in groups:
            for dep in deps:
                # strict=False: a genuine cycle in someone else's lock is skipped, not fatal.
                kg.assert_edge(name, "depends_on", dep["name"], source=source_tag)
                if "marker" in dep:
                    kg.update_node(dep["name"], marker=dep["marker"])
    return kg


def project_of(kg: KnowledgeGraph) -> str:
    """The node the lockfile was resolved for."""
    for node in kg.nodes:
        if node.type == "Project":
            return node.id
    raise ValueError("no Project node in the graph -- is this a uv.lock?")


def goal_for(kg: KnowledgeGraph) -> Goal:
    project = project_of(kg)
    return Goal(project, f"upgrade {project}'s dependency tree and resync")


# ---------------------------------------------------------------- the agent


def scripted_llm() -> ScriptedLLM:
    """What a model would extract from each observation -- traps included."""
    return ScriptedLLM(
        # Asked which ready step goes first, the model front-loads the slow wheel
        # download; the planner alone would have gone alphabetically (colorama first).
        action=lambda plan, context: "upgrade_package(pluggy)",
        claims_by_observation={
            "pygments is current": claims("pygments pinned_to pygments==2.21.0"),
            "pytest upgraded": claims(
                "pytest depends_on packaging",
                # Not in the lockfile at all: the policy refuses to invent a package.
                "pytest depends_on tomli",
                # The changelog summary got the arrow backwards.
                "packaging depends_on pytest",
            ),
        },
        dependencies_by_node={
            "python": claims(
                "python satisfied_by cpython-3.13",
                # An interpreter does not depend on a project: range violation.
                "python depends_on kg-agent",
            ),
        },
    )


def build_agent(kg: KnowledgeGraph, llm=None) -> KGAgent:
    agent = KGAgent(kg, llm or scripted_llm(), actions=DEPS_ACTIONS, execute="stage",
                    policy=IngestPolicy(accept_unknown=True, unknown_confidence=0.6,
                                        allow_new_entities=False))

    @agent.action("upgrade_package")
    def upgrade_package(agent: KGAgent, step) -> ActionResult | str:
        node = agent.kg.get_node(step.node)
        version = node.attrs.get("version", "?") if node else "?"
        if step.node == "pluggy":
            # Whether this is a retry is read off the world model, not off a
            # counter in this process -- so it survives the restart in section 4.
            if node is None or node.attrs.get("status") != "failed":
                return ActionResult(step, False,
                                    "pluggy wheel download timed out (pypi.org, 30s)",
                                    {"retry": True})
            return f"pluggy upgraded: pluggy pinned_to pluggy=={version}"
        if step.node == "pytest":
            return (f"pytest upgraded to {version}. Changelog notes pytest depends_on packaging "
                    f"and pytest depends_on tomli; packaging depends_on pytest.")
        return f"{step.node} is current at {version}: {step.node} pinned_to {step.node}=={version}"

    @agent.action("resync_project")
    def resync_project(agent: KGAgent, step) -> str:
        return f"uv sync complete: {step.node} environment rebuilt"

    return agent


# -------------------------------------------------------------- the sections


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def demo_world_model(kg: KnowledgeGraph, lock_path: Path) -> None:
    banner("1. The lockfile is the world model (parsed, not typed in)")
    project = project_of(kg)
    print(f"source: {lock_path}")
    print(f"{kg} -- every node and edge below came out of that file\n")
    print(kg.context_for(project, hops=2))
    print("\nThis rendering -- rebuilt from the graph on every turn -- is what the model sees.")


def demo_planning(kg: KnowledgeGraph, llm=None) -> None:
    banner("2. The graph plans the upgrade, and vets what the model proposes")
    goal = goal_for(kg)
    print(plan_for(kg, goal, actions=DEPS_ACTIONS).render())
    print("\nStages follow dependency depth: leaves first, then what needs them, then the project.")
    print("No action satisfies a Runtime, so python is reported blocked rather than guessed at.")

    agent = build_agent(kg, llm)
    print(f"\nclaim extraction and step choice by: {type(agent.llm).__name__}"
          + (f" ({agent.llm.model})" if hasattr(agent.llm, "model") else ""))
    print("Each turn acts on the whole ready frontier (execute='stage'). With five packages")
    print("ready at once the model is asked which goes first; only a step on the plan is taken.")
    print("\n--- running the loop: observe -> update -> query -> plan -> choose -> act -> observe ---")
    run = agent.run(goal)
    print(run.render())
    first = run.records[0]
    print(f"\nturn 1 opened with {first.step} ({first.chosen_by}'s choice); it failed, so it fell")
    print("into turn 2's frontier and was retried off the graph's status=failed, not a counter.")

    banner("...how the blocked interpreter got unblocked")
    print("Hitting the blocked leaf, the agent asked the model what python needs.")
    print("Two proposals came back; the graph took one and refused the other:")
    edge = next(iter(kg.match("python", "satisfied_by", None)), None)
    print(f"  accepted: {edge} (source={edge.source}, confidence={edge.confidence:.2f})"
          if edge else "  accepted: none (the model proposed nothing the graph already knew)")
    print(f"  refused : python depends_on kg-agent -- in the graph? "
          f"{kg.has_edge('python', 'depends_on', 'kg-agent')} "
          f"(kg-agent is a Project; depends_on requires a Package)")

    print("\nreplan after the run:")
    print(plan_for(kg, goal, actions=DEPS_ACTIONS).render())
    print(f"\nmessage history retained: 0 turns (the upgrade lives in the graph, rev {kg.revision})")
    for resolution in run.resolutions:
        print(f"contradiction {resolution.decision}: {resolution.verdict.claim} -- {resolution.reason}")


def demo_verification(kg: KnowledgeGraph) -> None:
    banner("3. The graph constrains what the agent is willing to believe")
    cases = [
        ("asserted by the lockfile", Claim.parse("pytest depends_on packaging")),
        ("derived: a transitive chain", Claim.parse("kg-agent depends_on pluggy")),
        ("the inverse, materialised on assert", Claim.parse("packaging required_by pytest")),
        ("the arrow points the other way", Claim.parse("packaging depends_on pytest")),
        ("a package the lock never resolved", Claim.parse("pytest depends_on tomli")),
    ]
    for label, claim in cases:
        verdict = verify(kg, claim)
        print(f"\n{label}:")
        print(f"  claim   : {claim}")
        print(f"  status  : {verdict.status.upper()}")
        print(f"  because : {verdict.explanation}")
        for edge in verdict.evidence:
            print(f"  evidence: {edge}")

    stale = Claim.parse("pytest pinned_to pytest==8.4.1")
    print("\na stale version, two layers deep:")
    print(f"  claim   : {stale}")
    print(f"  status  : {verify(kg, stale).status.upper()} -- {verify(kg, stale).explanation}")
    lenient = verify(kg, stale, require_known_entities=False)
    print(f"  ...and if new entities were allowed: {lenient.status.upper()} -- {lenient.explanation}")

    banner("...and the gate that keeps the world model clean")
    report = ingest(kg, [c for _, c in cases] + [stale],
                    policy=IngestPolicy(accept_unknown=True, allow_new_entities=False))
    print(report.summary())
    for verdict in report.rejected:
        print(f"  refused: {verdict}")
    for edge in report.written:
        print(f"  written provisionally: {edge} (confidence {edge.confidence:.2f})")
    pin = next(iter(kg.match("pytest", "pinned_to", None)), None)
    print(f"\npytest is still pinned where the lockfile put it: {pin}; "
          f"tomli never entered the graph: {not kg.has_node('tomli')}")


def demo_restart(lock_path: Path, llm=None) -> None:
    banner("4. The upgrade survives a restart (the graph is the whole handoff)")
    kg = load_lock(lock_path)
    goal = goal_for(kg)
    first = build_agent(kg, llm)
    partial = first.run(goal, max_steps=4)
    print(f"process 1 stopped mid-stage after {len(partial.records)} acted steps: {partial.reason}")
    print(partial.render())

    path = Path(tempfile.gettempdir()) / "kg_agent_deps.json"
    kg.save(path)
    print(f"\nsaved to {path}")

    resumed = KnowledgeGraph.load(path, ontology=lock_ontology())
    failed = sorted(n.id for n in resumed.nodes if n.attrs.get("status") == "failed")
    print(f"process 2 loaded a cold graph: {resumed} -- no transcript, no in-memory retry counters")
    print(f"carried over as failed: {', '.join(failed) or 'nothing'}")
    print(plan_for(resumed, goal, actions=DEPS_ACTIONS).render())

    second = build_agent(resumed, llm)
    finished = second.run(goal)
    print(f"\nprocess 2 finished the upgrade: completed={finished.completed} ({finished.reason})")
    print(finished.render())
    print("\nThe retry was not run blind: process 2 read pluggy's failed status off the graph.")


def main(argv: list[str] | None = None) -> None:
    """``--lock PATH`` picks the lockfile; ``--live`` swaps in OpenRouter."""
    argv = sys.argv[1:] if argv is None else argv
    lock_path = REPO_LOCK
    for i, arg in enumerate(argv):
        if arg == "--lock" and i + 1 < len(argv):
            lock_path = Path(argv[i + 1])
        elif arg.startswith("--lock="):
            lock_path = Path(arg.split("=", 1)[1])
    kg = load_lock(lock_path)
    llm = None
    if "--live" in argv:
        from .openrouter import OpenRouterLLM
        model = next((a.split("=", 1)[1] for a in argv if a.startswith("--model=")), None)
        llm = OpenRouterLLM(model, ontology=kg.ontology)
    demo_world_model(kg, lock_path)
    demo_planning(kg, llm)
    demo_verification(kg)
    demo_restart(lock_path, llm)
    if llm is not None:
        print(f"\nOpenRouter usage: {llm.usage}")
    print()


if __name__ == "__main__":
    main()
