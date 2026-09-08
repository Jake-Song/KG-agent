import json
import subprocess
import sys

import pytest

from kg_agent.research import ResearchSession, session_lock
from kg_agent.research_domain import ResearchError, ResearchOntology
from kg_agent import research_experiment as experiment


def cli(directory, command, *args, code=0):
    process = subprocess.run([sys.executable, "-m", "kg_agent.research", command,
                              "--dir", str(directory), *args], capture_output=True, text=True)
    assert process.returncode == code, process.stdout + process.stderr
    assert not process.stderr
    response = json.loads(process.stdout)
    assert response["ok"] == (code == 0)
    return response.get("result", response.get("error"))


def proposal(ident="quadratic", degree=2, penalty=0):
    return {"id": ident, "hypothesis": "Curvature explains the linear residual error",
            "rationale": "Compare a quadratic model against the measured linear baseline",
            "degree": degree, "penalty": penalty}


def submit(directory, p, code=0):
    path = directory / "proposal.json"
    path.write_text(json.dumps(p))
    return cli(directory, "propose", "--file", str(path), code=code)


def initialized(tmp_path, budget=5):
    session = ResearchSession.initialize(tmp_path, budget)
    session.run()
    return session


def test_full_cycle_across_processes(tmp_path):
    assert cli(tmp_path, "init", "--budget", "3")["next_commands"] == ["status", "run"]
    baseline = cli(tmp_path, "run")["trial"]
    assert "test_mse" not in (tmp_path / "state.json").read_text()
    submit(tmp_path, proposal())
    better = cli(tmp_path, "run")["trial"]
    assert better["validation_mse"] < baseline["validation_mse"] * 0.1
    assert better["conclusion"] == "observed_improvement"
    submit(tmp_path, proposal("heavy-ridge", 2, 1))
    worse = cli(tmp_path, "run")["trial"]
    assert worse["comparison_id"] == "quadratic"
    assert worse["conclusion"] == "no_observed_improvement"
    assert cli(tmp_path, "status")["next_commands"] == ["status", "finalize"]
    assert submit(tmp_path, proposal("cubic", 3), code=2)["code"] == "budget_exhausted"
    report = cli(tmp_path, "finalize")
    assert report["winner_id"] == "quadratic"
    assert report["test_mse"] >= 0
    assert report["unused_budget"] == 0
    before = (tmp_path / "state.json").read_bytes()
    assert cli(tmp_path, "finalize") == report
    assert (tmp_path / "state.json").read_bytes() == before
    assert submit(tmp_path, proposal("cubic", 3), code=2)["code"] == "finalized"
    assert cli(tmp_path, "run", code=2)["code"] == "finalized"
    assert "heavy-ridge" in (tmp_path / "report.md").read_text()
    session = ResearchSession.load(tmp_path)
    assert session.kg.ontology.knows("assesses")
    assert session.kg.get_edge("evaluation:quadratic", "evaluates", "model:quadratic").source == "evaluator"


def test_proposal_idempotency_and_pending_guards(tmp_path):
    cli(tmp_path, "init")
    assert submit(tmp_path, proposal(), code=2)["code"] == "baseline_required"
    assert cli(tmp_path, "finalize", code=2)["code"] == "baseline_required"
    cli(tmp_path, "run")
    submit(tmp_path, proposal())
    before = (tmp_path / "state.json").read_bytes()
    submit(tmp_path, proposal())
    assert (tmp_path / "state.json").read_bytes() == before
    assert submit(tmp_path, proposal(degree=3), code=2)["code"] == "id_conflict"
    assert submit(tmp_path, proposal("other", 3), code=2)["code"] == "pending_trial"
    assert cli(tmp_path, "finalize", code=2)["code"] == "pending_trial"
    cli(tmp_path, "run")
    assert submit(tmp_path, proposal("duplicate"), code=2)["code"] == "duplicate_configuration"
    assert cli(tmp_path, "run", code=2)["code"] == "no_pending_trial"
    assert cli(tmp_path, "init", code=2)["code"] == "session_exists"


def test_generate_previews_deterministic_next_hypothesis_without_mutation(tmp_path):
    cli(tmp_path, "init", "--budget", "3")
    assert cli(tmp_path, "generate", code=2)["code"] == "baseline_required"
    cli(tmp_path, "run")
    before = (tmp_path / "state.json").read_bytes()
    first = cli(tmp_path, "generate")
    second = cli(tmp_path, "generate")
    assert first == second
    assert first["based_on_trial_id"] == "baseline"
    assert first["proposal"]["degree"] == 2
    assert first["proposal"]["penalty"] == 0
    assert first["state"]["next_commands"] == ["status", "generate", "propose", "finalize"]
    assert (tmp_path / "state.json").read_bytes() == before


def test_generate_follows_validation_winner_and_excludes_failed_configs(tmp_path):
    session = initialized(tmp_path, budget=5)
    session.propose(proposal())
    session.run()
    generated = session.generate_hypothesis()
    assert generated["based_on_trial_id"] == "quadratic"
    assert generated["proposal"]["degree"] == 3
    assert generated["proposal"]["penalty"] == 0
    session.propose(generated["proposal"])
    session.run()
    next_proposal = session.generate_hypothesis()["proposal"]
    assert (next_proposal["degree"], next_proposal["penalty"]) == (4, 0)
    assert "current validation winner" in next_proposal["rationale"]


def test_generate_handles_id_collision_and_exhaustion(tmp_path):
    session = initialized(tmp_path, budget=3)
    session.propose({**proposal(), "id": "degree-3-penalty-0p001", "penalty": 0.001})
    session.run()
    generated = session.generate_hypothesis()["proposal"]
    assert generated["id"] == "degree-3-penalty-0p001-2"
    session.propose(generated)
    session.run()
    assert session.status()["next_commands"] == ["status", "finalize"]
    assert cli(tmp_path, "generate", code=2)["code"] == "budget_exhausted"


@pytest.mark.parametrize("change", [{"degree": True}, {"degree": 6}, {"penalty": float("nan")},
                                    {"penalty": -1}, {"id": "../outside"}, {"rationale": " "},
                                    {"validation_mse": 0}, {"penalty": True}])
def test_invalid_proposals_do_not_mutate(tmp_path, change):
    cli(tmp_path, "init")
    cli(tmp_path, "run")
    before = (tmp_path / "state.json").read_bytes()
    assert submit(tmp_path, proposal() | change, code=2)["code"] == "invalid_proposal"
    assert (tmp_path / "state.json").read_bytes() == before


def test_known_polynomial_and_ridge():
    rows = [(x / 10, 1 + 2*(x / 10) - 3*(x / 10)**2) for x in range(-10, 11)]
    coefficients = experiment.fit(rows, 2, 0)
    assert coefficients == pytest.approx([1, 2, -3], abs=1e-10)
    assert experiment.mse(rows, coefficients) < 1e-20
    constant = [(x / 10, 4.0) for x in range(-10, 11)]
    assert experiment.fit(constant, 2, 1) == pytest.approx([4, 0, 0], abs=1e-10)
    assert abs(experiment.fit(rows, 2, 1)[2]) < abs(coefficients[2])


def test_splits_fixed_and_test_only_at_finalization(tmp_path, monkeypatch):
    assert experiment.dataset(42, "train") == experiment.dataset(42, "train")
    assert experiment.dataset(42, "train") != experiment.dataset(43, "train")
    sets = [set(experiment.dataset(42, split)) for split in ("train", "validation", "test")]
    assert not sets[0] & sets[1] and not sets[1] & sets[2] and not sets[0] & sets[2]
    original = experiment.dataset
    calls = []
    def tracked(seed, split):
        calls.append(split)
        return original(seed, split)
    monkeypatch.setattr(experiment, "dataset", tracked)
    session = initialized(tmp_path)
    session.propose(proposal())
    session.run()
    session.status()
    assert "test" not in calls
    session.finalize()
    ResearchSession.load(tmp_path).finalize()
    assert calls.count("test") == 1


def test_interrupt_before_commit_resumes_same_trial(tmp_path, monkeypatch):
    session = ResearchSession.initialize(tmp_path)
    original = experiment.mse
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(experiment, "mse", interrupt)
    with pytest.raises(KeyboardInterrupt):
        session.run()
    recovered = ResearchSession.load(tmp_path)
    assert recovered.state["trials"][0]["status"] == "pending"
    assert not recovered.kg.has_node("model:baseline")
    monkeypatch.setattr(experiment, "mse", original)
    recovered.run()
    assert len(recovered.state["trials"]) == 1
    assert recovered.status()["remaining_budget"] == 4


def test_numerical_failure_is_recorded_and_baseline_blocks(tmp_path, monkeypatch):
    session = ResearchSession.initialize(tmp_path)
    monkeypatch.setattr(experiment, "fit", lambda *args: [float("inf"), 1])
    with pytest.raises(ResearchError, match="finite"):
        session.run()
    restored = ResearchSession.load(tmp_path)
    assert restored.state["trials"][0]["status"] == "failed"
    assert restored.status()["next_commands"] == ["status"]
    assert cli(tmp_path, "finalize", code=2)["code"] == "baseline_required"


def test_failed_followup_consumes_slot_but_allows_finalize(tmp_path, monkeypatch):
    session = initialized(tmp_path, budget=2)
    session.propose(proposal())
    monkeypatch.setattr(experiment, "fit", lambda *args: [0])
    with pytest.raises(ResearchError):
        session.run()
    restored = ResearchSession.load(tmp_path)
    assert restored.status()["remaining_budget"] == 0
    assert restored.finalize()["winner_id"] == "baseline"


def test_early_finalize_report_repair_and_tie(tmp_path):
    session = initialized(tmp_path)
    session.propose(proposal())
    session.run()
    # Ranking stability independent of trial quality: exact ties favor earlier trials.
    session.state["trials"][1]["validation_mse"] = session.state["trials"][0]["validation_mse"]
    assert session.best()["proposal"]["id"] == "baseline"
    report = session.finalize()
    assert report["unused_budget"] == 3
    (tmp_path / "report.md").unlink()
    assert ResearchSession.load(tmp_path).finalize() == report
    assert (tmp_path / "report.md").exists()


def test_machine_errors_and_lock(tmp_path):
    assert cli(tmp_path, "status", code=2)["code"] == "session_missing"
    assert cli(tmp_path, "init", "--budget", "0", code=2)["code"] == "invalid_configuration"
    cli(tmp_path, "init")
    with session_lock(tmp_path):
        assert cli(tmp_path, "run", code=3)["code"] == "session_busy"
    cli(tmp_path, "run")
    state = json.loads((tmp_path / "state.json").read_text())
    state["ontology_version"] = 999
    (tmp_path / "state.json").write_text(json.dumps(state))
    assert cli(tmp_path, "status", code=2)["code"] == "unsupported_version"


def test_contract_transition_requires_expected_type_and_state():
    from kg_agent import Node
    onto = ResearchOntology()
    with pytest.raises(ResearchError):
        onto.check_action("fit", Node("x", "Experiment", {"status": "complete"}))
    with pytest.raises(ResearchError):
        onto.check_action("evaluate", Node("x", "Hypothesis", {"status": "pending"}))


def test_argument_and_json_errors_are_machine_readable(tmp_path):
    assert cli(tmp_path, "init", "--budget", "bad", code=2)["code"] == "invalid_arguments"
    cli(tmp_path, "init")
    path = tmp_path / "bad.json"
    path.write_text('{')
    assert cli(tmp_path, "propose", "--file", str(path), code=2)["code"] == "invalid_json"
    path.write_text('[]')
    assert cli(tmp_path, "propose", "--file", str(path), code=2)["code"] == "invalid_proposal"


def test_failed_metric_never_completes_trial_and_returns_execution_error(tmp_path, monkeypatch, capsys):
    from kg_agent.research import main
    ResearchSession.initialize(tmp_path)
    monkeypatch.setattr(experiment, "mse", lambda *args: float("nan"))
    assert main(["run", "--dir", str(tmp_path)]) == 3
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "experiment_failed"
    session = ResearchSession.load(tmp_path)
    assert session.state["trials"][0]["status"] == "failed"
    assert session.kg.get_node("evaluation:baseline").attrs["status"] == "failed"
    assert "validation_mse" not in session.state["trials"][0]


def test_atomic_failure_preserves_previous_state(tmp_path, monkeypatch):
    import kg_agent.research as research
    session = ResearchSession.initialize(tmp_path)
    before = (tmp_path / "state.json").read_bytes()
    def fail_replace(*args):
        raise OSError("Simulated storage failure")
    monkeypatch.setattr(research.os, "replace", fail_replace)
    with pytest.raises(OSError):
        session.run()
    assert (tmp_path / "state.json").read_bytes() == before
    assert not list(tmp_path.glob('.state.json-*'))
    assert ResearchSession.load(tmp_path).state["trials"][0]["status"] == "pending"


def test_step_explanations_follow_evidence_without_mutating_state(tmp_path):
    session = ResearchSession.initialize(tmp_path)
    before = (tmp_path / "state.json").read_bytes()
    pending = cli(tmp_path, "status")
    steps = {s["step"]: s for s in pending["trials"][0]["steps"]}
    assert steps["fit"]["status"] == "pending"
    assert steps["evaluate"]["status"] == "pending"
    assert "Saved validation MSE" not in steps["evaluate"]["description"]
    assert (tmp_path / "state.json").read_bytes() == before
    cli(tmp_path, "run")
    submit(tmp_path, proposal())
    trial = cli(tmp_path, "run")["trial"]
    steps = {s["step"]: s for s in trial["steps"]}
    assert steps["fit"]["status"] == "complete"
    assert f"{trial['validation_mse']:.8g}" in steps["evaluate"]["description"]
    assert "Criterion met" in steps["compare"]["description"]
    submit(tmp_path, proposal("ridge", 2, 0.001))
    trial = cli(tmp_path, "run")["trial"]
    assert "Criterion not met" in next(s for s in trial["steps"] if s["step"] == "compare")["description"]
    cli(tmp_path, "finalize")
    before = (tmp_path / "state.json").read_bytes()
    cli(tmp_path, "finalize")
    assert (tmp_path / "state.json").read_bytes() == before
    report = (tmp_path / "report.md").read_text()
    assert "**fit (complete)**" in report
    assert "## Finalization" in report
    assert "not historical execution logs" in report
    assert "finalized" in cli(tmp_path, "status")["description"]


def test_failure_explanations_do_not_claim_evaluation(tmp_path, monkeypatch):
    session = ResearchSession.initialize(tmp_path)
    monkeypatch.setattr(experiment, "fit", lambda *args: [float('inf'), 1])
    with pytest.raises(ResearchError):
        session.run()
    status = ResearchSession.load(tmp_path).status()
    steps = {s["step"]: s for s in status["trials"][0]["steps"]}
    assert steps["fit"]["status"] == "failed"
    assert steps["evaluate"]["status"] == "skipped"
    assert steps["compare"]["status"] == "skipped"
    assert "blocked" in status["description"]
