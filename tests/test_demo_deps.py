from pathlib import Path

from kg_agent import KnowledgeGraph, Status, plan_for
from kg_agent.demo_deps import (DEPS_ACTIONS, REPO_LOCK, build_agent, goal_for, load_lock,
                                lock_ontology, main)

MINI_LOCK = """\
version = 1
requires-python = ">=3.13"

[[package]]
name = "app"
version = "0.1.0"
source = { virtual = "." }
dependencies = [{ name = "web" }]

[package.dev-dependencies]
dev = [{ name = "pytest" }]

[[package]]
name = "web"
version = "2.0.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "colorama", marker = "sys_platform == 'win32'" },
    { name = "core" },
]

[[package]]
name = "core"
version = "1.4.2"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "colorama"
version = "0.4.6"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "pytest"
version = "9.1.1"
source = { registry = "https://pypi.org/simple" }
dependencies = [{ name = "core" }]
"""


def mini_lock(tmp_path: Path) -> Path:
    path = tmp_path / "uv.lock"
    path.write_text(MINI_LOCK)
    (tmp_path / ".python-version").write_text("3.13\n")
    return path


def test_load_lock_builds_dependency_graph(tmp_path):
    kg = load_lock(mini_lock(tmp_path))

    assert kg.node_type("app") == "Project"
    assert kg.node_type("web") == "Package"
    assert kg.node_type("web==2.0.0") == "Version"
    assert kg.node_type("python") == "Runtime"
    assert kg.node_type("cpython-3.13") == "Interpreter"
    assert kg.node_type("pypi.org") == "Registry"

    edge = kg.get_edge("web", "depends_on", "core")
    assert edge is not None and edge.source == "uv.lock"
    assert kg.get_edge("app", "depends_on", "pytest").source == "uv.lock:dev"
    assert kg.has_edge("core", "required_by", "web")          # inverse materialised
    assert kg.has_edge("web", "pinned_to", "web==2.0.0")
    assert kg.has_edge("app", "runs_on", "python")
    assert kg.get_node("colorama").attrs["marker"] == "sys_platform == 'win32'"
    assert kg.get_node("web").attrs["status"] == "locked"


def test_load_lock_on_the_repo_lockfile():
    kg = load_lock(REPO_LOCK)
    assert kg.has_edge("kg-agent", "depends_on", "pytest")
    assert kg.has_edge("pytest", "depends_on", "packaging")
    assert goal_for(kg).target == "kg-agent"


def test_plan_stages_follow_dependency_depth(tmp_path):
    kg = load_lock(mini_lock(tmp_path))
    plan = plan_for(kg, goal_for(kg), actions=DEPS_ACTIONS)
    stage = {step.node: step.stage for step in plan.steps}

    assert stage["core"] == stage["colorama"] == 0
    assert stage["web"] == stage["pytest"] == 1
    assert stage["app"] == 2
    assert plan.blocked == ["python"]


def test_reversed_dependency_is_contradicted_and_unknown_package_is_ill_formed(tmp_path):
    from kg_agent import Claim, verify

    kg = load_lock(mini_lock(tmp_path))
    assert verify(kg, Claim.parse("core depends_on web")).status is Status.CONTRADICTED
    assert verify(kg, Claim.parse("app depends_on core")).status is Status.ENTAILED
    assert verify(kg, Claim.parse("web depends_on tomli")).status is Status.ILL_FORMED


def test_agent_completes_and_rejects_traps():
    kg = load_lock(REPO_LOCK)
    run = build_agent(kg).run(goal_for(kg))

    assert run.completed
    assert [str(v.claim) for v in run.contradictions] == ["packaging depends_on pytest"]
    assert not kg.has_node("tomli")
    assert kg.has_edge("pytest", "pinned_to", "pytest==9.1.1")
    pluggy = [(r.step.node, r.result.ok) for r in run.records if r.step and r.step.node == "pluggy"]
    assert pluggy == [("pluggy", False), ("pluggy", True)]
    assert kg.has_edge("python", "satisfied_by", "cpython-3.13")
    assert not kg.has_edge("python", "depends_on", "kg-agent")


def test_restart_resumes_from_disk(tmp_path):
    kg = load_lock(REPO_LOCK)
    goal = goal_for(kg)
    partial = build_agent(kg).run(goal, max_steps=4)
    assert not partial.completed
    assert kg.get_node("pluggy").attrs["status"] == "failed"

    path = tmp_path / "world.json"
    kg.save(path)
    resumed = KnowledgeGraph.load(path, ontology=lock_ontology())
    assert resumed.get_node("pluggy").attrs["status"] == "failed"

    finished = build_agent(resumed).run(goal)
    assert finished.completed
    # The second process retried off the graph's status: no repeated failure.
    assert [r.step.node for r in finished.records if r.step][:1] == ["pluggy"]
    assert all(r.result.ok for r in finished.records if r.result)


def test_main_runs_offline(capsys):
    main([])
    out = capsys.readouterr().out
    for banner in ("1. The lockfile is the world model", "2. The graph plans the upgrade",
                   "3. The graph constrains", "4. The upgrade survives a restart"):
        assert banner in out
    assert "completed=True" in out


def test_stage_mode_reorders_the_frontier_by_the_models_choice():
    kg = load_lock(REPO_LOCK)
    run = build_agent(kg).run(goal_for(kg))
    assert run.completed
    assert run.records[0].step.node == "pluggy" and run.records[0].chosen_by == "model"
    assert {r.turn for r in run.records[:5]} == {1}
    assert [r.step.node for r in run.records[1:5]] == ["colorama", "iniconfig", "packaging",
                                                        "pygments"]
    assert run.records[5].step.node == "pluggy" and run.records[5].turn == 2
    acted = [r for r in run.records if r.step]
    assert [r.iteration for r in acted] == list(range(1, len(acted) + 1))
    assert [r.note for r in run.records if not r.step] == [
        "blocked on python; asked the model, which added 1 edge"]
    assert [str(v.claim) for v in run.unresolved] == ["packaging depends_on pytest"]
    assert "(model's choice)" in run.render()
