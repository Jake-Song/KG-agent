import json
from pathlib import Path

from kg_agent import Claim, KnowledgeGraph, Status, plan_for, verify
from kg_agent.demo_study import (SNAPSHOT, STUDY_ACTIONS, build_agent, goal_for, load_repo,
                                 main, study_ontology)

MINI_README = """\
# Mini Lab

첫 데이터 로드 시 원본을 내려받아 `data/Downloads/`에 저장합니다.

```bash
uv run python -m mini_lab.week1
uv run python -m mini_lab.week2
```
"""

MINI_PYPROJECT = """\
[project]
name = "mini-lab"
version = "0.2.0"
requires-python = ">=3.13"
"""

MINI_GITIGNORE = """\
data/
# Large regenerable table
results/week2/predictions.csv
logs/
"""

MINI_BUDGET = """\
regime,strategy,final_rmse,speedup,reaches_target
in_domain,random,0.740000,1.000000,True
in_domain,uncertainty,0.720000,1.050000,True
in_domain,diversity,0.790000,,False
out_of_scaffold,random,1.090000,1.000000,True
out_of_scaffold,uncertainty,1.110000,,False
out_of_scaffold,diversity,1.010000,1.710000,True
"""


def mini_repo(tmp_path: Path, *, with_source: bool = False, with_splits: bool = False) -> Path:
    (tmp_path / "README.md").write_text(MINI_README, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(MINI_PYPROJECT)
    (tmp_path / ".gitignore").write_text(MINI_GITIGNORE)
    week1, week2 = tmp_path / "results" / "week1", tmp_path / "results" / "week2"
    week1.mkdir(parents=True)
    week2.mkdir(parents=True)
    (week1 / "sweep_config.json").write_text(json.dumps({"seeds": [0, 1, 2],
                                                         "split_types": ["random", "scaffold"]}))
    (week2 / "acquisition_config.json").write_text(json.dumps(
        {"config": {"seeds": [0, 1], "split_types": ["random"], "architecture": "GIN",
                    "strategies": ["random", "uncertainty", "diversity"],
                    "headline_strategy": "uncertainty"}}))
    (week2 / "budget_efficiency.csv").write_text(MINI_BUDGET)
    (week2 / "summary.json").write_text(json.dumps(
        {"question": "does choosing help?", "caveats": ["the labels already exist"],
         "final_comparison": [{"regime": "out_of_scaffold", "strategy": "diversity",
                               "budget": 500, "excludes_zero": False}]}))
    if with_splits:
        (week1 / "splits.json").write_text(json.dumps({"random:0": {"train": [1, 2]}}))
    if with_source:
        package = tmp_path / "src" / "mini_lab"
        package.mkdir(parents=True)
        (package / "week1.py").write_text("VALUE = 1\n")
        (package / "week2.py").write_text("from .week1 import VALUE\n")
    return tmp_path


def test_load_repo_builds_the_study_graph(tmp_path):
    kg = load_repo(mini_repo(tmp_path))

    assert kg.node_type("mini-lab") == "Project"
    assert kg.node_type("week1") == kg.node_type("week2") == "Study"
    assert kg.node_type("random") == "Split"
    assert kg.node_type("esol") == "Dataset"
    assert kg.node_type("data/Downloads") == "LocalCopy"
    assert kg.node_type("GIN") == "Architecture"
    assert kg.node_type("strategy:uncertainty") == "Strategy"

    assert kg.get_edge("mini-lab", "requires", "week2").source == "README.md"
    assert kg.has_edge("week1", "evaluates", "scaffold")
    assert kg.has_edge("scaffold", "drawn_from", "esol")
    assert kg.has_edge("week2", "uses_architecture", "GIN")
    assert kg.has_edge("week2", "headline_strategy", "strategy:uncertainty")
    assert kg.get_node("week1").attrs["seeds"] == "0-2"

    # committed files become artifacts; the gitignored one is what keeps week2 unsatisfied
    assert kg.has_edge("week2", "produced", "results/week2/budget_efficiency.csv")
    assert kg.get_node("results/week2/predictions.csv").attrs["status"] == "missing"
    assert kg.has_edge("week2", "must_produce", "results/week2/predictions.csv")
    assert kg.get_node("week1").attrs.get("status") == "complete"   # nothing of week1's is missing
    assert kg.get_node("week2").attrs.get("status") is None


def test_rankings_are_regime_scoped_and_only_adjacent_pairs_are_asserted(tmp_path):
    kg = load_repo(mini_repo(tmp_path))

    # in domain the order is uncertainty, random, diversity; out of scaffold it reverses
    assert kg.has_edge("in_domain:uncertainty", "outperforms", "in_domain:random")
    assert kg.has_edge("out_of_scaffold:diversity", "outperforms", "out_of_scaffold:random")
    assert kg.has_edge("out_of_scaffold:random", "outperforms", "out_of_scaffold:uncertainty")
    # non-adjacent pairs are derived, never stored
    assert not kg.has_edge("in_domain:uncertainty", "outperforms", "in_domain:diversity")
    assert verify(kg, Claim.parse(
        "in_domain:uncertainty outperforms in_domain:diversity")).status is Status.ENTAILED
    assert kg.get_node("out_of_scaffold:diversity").attrs["speedup"] == "1.710000"
    assert kg.has_edge("out_of_scaffold:diversity", "measures", "strategy:diversity")


def test_committed_splits_and_source_imports_change_the_plan(tmp_path):
    plain = load_repo(mini_repo(tmp_path))
    assert plan_for(plain, goal_for(plain), actions=STUDY_ACTIONS).blocked == ["esol"]

    rich = tmp_path / "rich"
    rich.mkdir()
    kg = load_repo(mini_repo(rich, with_source=True, with_splits=True))
    assert kg.get_node("random").attrs["status"] == "materialised"
    assert kg.has_edge("week2", "derived_from", "week1")   # parsed from the module imports
    plan = plan_for(kg, goal_for(kg), actions=STUDY_ACTIONS)
    # the dataset sits under satisfied splits, so it never enters the plan; week1 owns
    # nothing gitignored, so it is pruned as already satisfied
    assert plan.blocked == []
    assert "week1" in plan.satisfied
    stage = {step.node: step.stage for step in plan.steps}
    assert stage == {"week2": 0, "mini-lab": 1}


def test_load_repo_on_the_bundled_snapshot():
    kg = load_repo(SNAPSHOT)

    assert sorted(n.id for n in kg.nodes if n.type == "Study") == ["week3", "week4", "week5",
                                                                  "week6"]
    assert kg.has_edge("ai4sci-molecule", "requires", "week6")
    assert kg.has_edge("week4", "evaluates", "scaffold_shuffled")   # from its regime map
    assert kg.has_edge("week6", "uses_architecture", "GIN")
    assert kg.get_node("esol").attrs == {"source": "MoleculeNet", "molecules": 1128,
                                         "scaffolds": 269}
    missing = sorted(e.object for e in kg.match(None, "must_produce", None))
    assert missing == ["results/week3/predictions.csv", "results/week4/molecule_errors.csv",
                       "results/week5/member_predictions.csv",
                       "results/week6/test_predictions.csv"]
    assert len(list(kg.match("week6", "limits", None))) == 3


def test_plan_stages_follow_dependency_depth():
    kg = load_repo(SNAPSHOT)
    plan = plan_for(kg, goal_for(kg), actions=STUDY_ACTIONS)
    stage = {step.node: step.stage for step in plan.steps}

    assert plan.blocked == ["esol"]
    assert stage["esol"] == 0
    assert stage["random"] == stage["scaffold"] == stage["scaffold_shuffled"] == 1
    assert stage["week3"] == stage["week4"] == stage["week5"] == stage["week6"] == 2
    assert stage["ai4sci-molecule"] == 3
    assert plan.ready == []          # the only stage-0 step is blocked


def test_the_findings_constrain_what_can_be_believed():
    kg = load_repo(SNAPSHOT)
    verdict = verify(kg, Claim.parse(
        "out_of_scaffold:uncertainty outperforms out_of_scaffold:random"))

    assert verdict.status is Status.CONTRADICTED          # the repo's own headline
    assert verify(kg, Claim.parse(
        "out_of_scaffold:diversity outperforms out_of_scaffold:random")).status is Status.ENTAILED
    assert verify(kg, Claim.parse(
        "out_of_scaffold:bald outperforms out_of_scaffold:random")).status is Status.ILL_FORMED
    assert verify(kg, Claim.parse("week6 uses_architecture GCN")).status is Status.CONTRADICTED
    # in domain the same pair goes the other way, which is why arms are regime-scoped
    assert verify(kg, Claim.parse(
        "in_domain:uncertainty outperforms in_domain:random")).status is Status.SUPPORTED


def test_agent_completes_and_rejects_traps():
    kg = load_repo(SNAPSHOT)
    run = build_agent(kg).run(goal_for(kg))

    assert run.completed
    assert [str(v.claim) for v in run.contradictions] == [
        "out_of_scaffold:GIN outperforms out_of_scaffold:MLP",
        "week3 derived_from week4",
        "out_of_scaffold:uncertainty outperforms out_of_scaffold:random",
    ]
    assert not kg.has_node("out_of_scaffold:bald")
    assert kg.has_edge("esol", "satisfied_by", "data/MoleculeNet")
    assert not kg.has_edge("esol", "evaluates", "random")
    # the run learned the study order the committed tables never state
    assert kg.get_edge("week4", "derived_from", "week3").source == "llm"
    assert not kg.has_edge("week3", "derived_from", "week4")
    week6 = [(r.step.node, r.result.ok) for r in run.records if r.step and r.step.node == "week6"]
    assert week6 == [("week6", False), ("week6", True)]
    assert kg.get_node("results/week6/test_predictions.csv").attrs["status"] == "rebuilt"


def test_stage_mode_reorders_the_frontier_by_the_models_choice():
    kg = load_repo(SNAPSHOT)
    run = build_agent(kg).run(goal_for(kg))

    assert [r.note for r in run.records if not r.step] == [
        "blocked on esol; asked the model, which added 1 edge"]
    acted = [r for r in run.records if r.step]
    assert [r.step.node for r in acted[:4]] == ["random", "scaffold", "scaffold_shuffled", "week6"]
    assert acted[3].chosen_by == "model" and acted[3].turn == 3
    assert [r.step.node for r in acted[4:7]] == ["week3", "week4", "week5"]
    assert acted[7].step.node == "week6" and acted[7].turn == 4
    assert [r.iteration for r in acted] == list(range(1, len(acted) + 1))
    assert "(model's choice)" in run.render()


def test_restart_resumes_from_disk(tmp_path):
    kg = load_repo(SNAPSHOT)
    goal = goal_for(kg)
    partial = build_agent(kg).run(goal, max_steps=6)
    assert not partial.completed
    assert kg.get_node("week6").attrs["status"] == "failed"

    path = tmp_path / "world.json"
    kg.save(path)
    resumed = KnowledgeGraph.load(path, ontology=study_ontology())
    assert resumed.get_node("week6").attrs["status"] == "failed"

    finished = build_agent(resumed).run(goal)
    assert finished.completed
    # The second process retried off the graph's status: no repeated failure.
    assert [r.step.node for r in finished.records if r.step][:1] == ["week6"]
    assert all(r.result.ok for r in finished.records if r.result)


def test_main_runs_offline(capsys):
    main([])
    out = capsys.readouterr().out
    for banner in ("1. A research repository is the world model", "2. The graph plans the rebuild",
                   "3. The graph constrains", "4. The rebuild survives a restart"):
        assert banner in out
    assert "completed=True" in out


def test_main_accepts_a_repo_path(tmp_path, capsys):
    """The sections narrate whatever the checkout holds, without assuming this project."""
    main(["--repo", str(mini_repo(tmp_path, with_source=True))])
    out = capsys.readouterr().out
    assert "mini-lab" in out
    assert "4. The rebuild survives a restart" in out
