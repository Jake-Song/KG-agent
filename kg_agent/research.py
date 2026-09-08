"""Persistent research commands for external coding agents.

Run ``python -m kg_agent.research --help``. Metrics and models are produced only
by the fixed evaluator; proposal text is retained as a hypothesis, not evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

from .graph import KnowledgeGraph
from .planner import plan_for
from .research_domain import ResearchError, ResearchOntology
from .research_explain import COMMAND_DESCRIPTIONS, explained_trial, next_step_description, trial_steps
from . import research_experiment as experiment

FORMAT_VERSION = 1


def atomic_write(path, text):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def session_lock(directory):
    # flock is released by the OS on process death; no stale lock recovery needed.
    with (directory / ".research.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ResearchError("session_busy", "Another process is using this session") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class ResearchSession:
    def __init__(self, directory, state):
        self.directory = Path(directory)
        self.state = state
        self.ontology = ResearchOntology()
        self.kg = KnowledgeGraph.from_json(state["graph"], ontology=self.ontology)

    @classmethod
    def load(cls, directory):
        path = Path(directory) / "state.json"
        if not path.exists():
            raise ResearchError("session_missing", "Initialize the session first")
        try:
            state = json.loads(path.read_text())
            if state["format_version"] != FORMAT_VERSION or state["ontology_version"] != ResearchOntology.version:
                raise ResearchError("unsupported_version", "Unsupported session or ontology version")
            return cls(directory, state)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ResearchError("invalid_state", f"Cannot read session: {exc}") from exc

    @classmethod
    def initialize(cls, directory, budget=5, seed=42):
        if (Path(directory) / "state.json").exists():
            raise ResearchError("session_exists", "Refusing to overwrite an existing session")
        if type(budget) is not int or not 1 <= budget <= 25 or type(seed) is not int:
            raise ResearchError("invalid_configuration", "Budget must be 1-25 and seed must be an integer")
        kg = KnowledgeGraph(ontology=ResearchOntology())
        kg.add_node("study", "Study", status="active")
        kg.add_node("dataset", "Dataset", status="complete", seed=seed,
                    benchmark_version=1, splits={"train": 80, "validation": 40, "test": 40})
        state = {"format_version": FORMAT_VERSION, "ontology_version": ResearchOntology.version,
                 "status": "active", "budget": budget, "seed": seed, "trials": [],
                 "graph": kg.to_json(), "final_report": None}
        session = cls(directory, state)
        session.queue({"id": "baseline", "hypothesis": "Establish the linear baseline",
                       "rationale": "Reference for subsequent validation comparisons",
                       "degree": 1, "penalty": 0}, None)
        session.save()
        return session

    def save(self):
        self.state["graph"] = self.kg.to_json()
        atomic_write(self.directory / "state.json", json.dumps(self.state, indent=2, allow_nan=False) + "\n")

    def best(self):
        successful = [t for t in self.state["trials"] if t["status"] == "complete"]
        return min(successful, key=lambda t: t["validation_mse"], default=None)

    def queue(self, proposal, comparison):
        ident = proposal["id"]
        trial = {"proposal": dict(proposal), "status": "pending", "comparison_id": comparison,
                 "source": "baseline" if comparison is None else "agent"}
        self.state["trials"].append(trial)
        self.kg.add_node(f"hypothesis:{ident}", "Hypothesis", text=proposal["hypothesis"],
                         rationale=proposal["rationale"], status="proposed")
        self.kg.add_node(f"experiment:{ident}", "Experiment", status="pending",
                         degree=proposal["degree"], penalty=proposal["penalty"])
        self.kg.add_node(f"evaluation:{ident}", "Evaluation", status="pending")
        edges = [("study", "contains", f"experiment:{ident}"),
                 (f"experiment:{ident}", "tests", f"hypothesis:{ident}"),
                 (f"experiment:{ident}", "uses_dataset", "dataset"),
                 (f"evaluation:{ident}", "requires", f"experiment:{ident}"),
                 (f"evaluation:{ident}", "assesses", f"hypothesis:{ident}")]
        if comparison:
            edges.append((f"hypothesis:{ident}", "compared_with", f"experiment:{comparison}"))
        for triple in edges:
            self.kg.assert_edge(*triple, strict=True, source=trial["source"])
        return trial

    def propose(self, proposal):
        self.ontology.validate_proposal(proposal)
        for trial in self.state["trials"]:
            if trial["proposal"]["id"] == proposal["id"]:
                if trial["proposal"] == proposal:
                    return trial
                raise ResearchError("id_conflict", "Proposal ID already has different content")
        self.ontology.check_command("propose", self.state)
        if any((t["proposal"]["degree"], t["proposal"]["penalty"]) ==
               (proposal["degree"], proposal["penalty"]) for t in self.state["trials"]):
            raise ResearchError("duplicate_configuration", "This configuration has already been proposed")
        trial = self.queue(proposal, self.best()["proposal"]["id"])
        self.save()
        return trial

    def run(self):
        self.ontology.check_command("run", self.state)
        trial = next(t for t in self.state["trials"] if t["status"] == "pending")
        proposal = trial["proposal"]
        ident, degree = proposal["id"], proposal["degree"]
        # No intermediate commit: interruption repeats a deterministic trial under
        # the same ID. Only complete model + evaluation evidence is committed.
        try:
            while True:
                plan = plan_for(self.kg, f"evaluation:{ident}", actions=self.ontology.action_map)
                if plan.complete:
                    break
                step = plan.ready[0]
                self.ontology.check_action(step.action, self.kg.get_node(step.node))
                if step.action == "fit":
                    coefficients = experiment.fit(experiment.dataset(self.state["seed"], "train"),
                                                  degree, proposal["penalty"])
                    self.ontology.check_evidence("fit", {"coefficients": coefficients}, degree)
                    trial["coefficients"] = coefficients
                    self.kg.add_node(f"model:{ident}", "Model", coefficients=coefficients, source="evaluator")
                    self.kg.assert_edge(step.node, "produced", f"model:{ident}", strict=True, source="evaluator")
                else:
                    score = experiment.mse(experiment.dataset(self.state["seed"], "validation"),
                                           trial["coefficients"])
                    self.ontology.check_evidence("evaluate", {"validation_mse": score}, degree)
                    trial["validation_mse"] = score
                    trial["evidence_source"] = "evaluator"
                    self.kg.update_node(step.node, validation_mse=score, source="evaluator")
                    self.kg.assert_edge(step.node, "evaluates", f"model:{ident}", strict=True, source="evaluator")
                self.kg.update_node(step.node, status=self.ontology.actions[step.action].completion_state)
            trial["status"] = "complete"
            comparison = next((t for t in self.state["trials"]
                               if t["proposal"]["id"] == trial["comparison_id"]), None)
            if comparison:
                reference = comparison["validation_mse"]
                improvement = (reference - trial["validation_mse"]) / reference if reference > 0 else 0.0
                trial["relative_improvement"] = improvement
                trial["conclusion"] = ("observed_improvement" if improvement >= self.ontology.improvement_threshold
                                       else "no_observed_improvement")
            else:
                trial["conclusion"] = "baseline_established"
            self.kg.update_node(f"hypothesis:{ident}", status="evaluated", conclusion=trial["conclusion"])
        except (ArithmeticError, ResearchError) as exc:
            trial["status"] = "failed"
            trial["error"] = {"code": getattr(exc, "code", "numerical_failure"), "message": str(exc)}
            self.kg.update_node(f"evaluation:{ident}", status="failed", error=trial["error"])
            if self.kg.get_node(f"experiment:{ident}").attrs["status"] != "complete":
                self.kg.update_node(f"experiment:{ident}", status="failed")
            self.save()
            raise ResearchError("experiment_failed", str(exc)) from exc
        self.save()
        return trial

    def status(self):
        allowed = []
        if self.state["status"] == "finalized":
            allowed = ["status", "finalize"]
        else:
            allowed = ["status"]
            for command in ("run", "propose", "finalize"):
                try:
                    self.ontology.check_command(command, self.state)
                except ResearchError:
                    continue
                allowed.append(command)
        best = self.best()
        return {"status": self.state["status"], "seed": self.state["seed"],
                "budget": self.state["budget"], "remaining_budget": self.state["budget"] - len(self.state["trials"]),
                "contracts": self.ontology.describe(),
                "trials": [explained_trial(t, self.ontology) for t in self.state["trials"]],
                "description": next_step_description(self.state),
                "best_trial_id": best["proposal"]["id"] if best else None,
                "next_commands": allowed, "final_report": self.state["final_report"]}

    def finalize(self):
        if self.state["status"] != "finalized":
            self.ontology.check_command("finalize", self.state)
            best = self.best()
            score = experiment.mse(experiment.dataset(self.state["seed"], "test"), best["coefficients"])
            self.ontology.check_evidence("evaluate", {"validation_mse": score}, best["proposal"]["degree"])
            self.state["final_report"] = {
                "winner_id": best["proposal"]["id"], "validation_mse": best["validation_mse"],
                "test_mse": score, "unused_budget": self.state["budget"] - len(self.state["trials"]),
                "interpretation": "Fixed synthetic benchmark; validation comparisons are not statistical proof.",
            }
            self.state["status"] = "finalized"
            self.kg.update_node("study", status="complete", **self.state["final_report"])
            self.save()
        self.write_report()
        return self.state["final_report"]

    def write_report(self):
        report = self.state["final_report"]
        lines = ["# Research report", "", f"Winner: {report['winner_id']}",
                 f"Validation MSE: {report['validation_mse']:.8g}", f"Test MSE: {report['test_mse']:.8g}",
                 f"Unused budget: {report['unused_budget']}", "", report["interpretation"], ""]
        lines.extend(["## How this cycle works", "",
                      "The ontology defines permitted configurations and evidence requirements. "
                      "The agent proposes hypotheses; the planner orders actions; the evaluator measures results.", "",
                      "Step descriptions below are derived from saved state and evidence, not historical execution logs.", ""])
        for trial in self.state["trials"]:
            p = trial["proposal"]
            lines.extend([f"## {p['id']}", "", p["hypothesis"], "", p["rationale"], "",
                          f"Degree: {p['degree']}; penalty: {p['penalty']}; status: {trial['status']}",
                          f"Validation MSE: {trial.get('validation_mse', 'unavailable')}",
                          f"Comparison: {trial['comparison_id']}; conclusion: {trial.get('conclusion', 'failed')}", ""])
            if "error" in trial:
                lines.extend([f"Failure: {trial['error']['message']}", ""])
            for number, step in enumerate(trial_steps(trial, self.ontology), 1):
                lines.extend([f"{number}. **{step['step']} ({step['status']})** — {step['description']}", ""])
        lines.extend(["## Finalization", "",
                      f"1. Select the smallest validation MSE across successful trials: {report['winner_id']}. "
                      "Exact ties favor the earlier trial.", "",
                      f"2. Evaluate that saved model on 40 test observations: MSE {report['test_mse']:.8g}. "
                      "The test score does not select the winner.", "",
                      "3. Save the final scores and freeze the study. Repeating finalize reads these scores; "
                      "it does not rerun experiments or test evaluation.", ""])
        atomic_write(self.directory / "report.md", "\n".join(lines))


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ResearchError("invalid_arguments", message)


def main(argv=None):
    try:
        parser = Parser(description=__doc__)
        subs = parser.add_subparsers(dest="command", required=True)
        for command in ("init", "status", "propose", "run", "finalize"):
            sub = subs.add_parser(command)
            sub.add_argument("--dir", required=True, type=Path)
            if command == "init":
                sub.add_argument("--budget", type=int, default=5)
                sub.add_argument("--seed", type=int, default=42)
            if command == "propose":
                sub.add_argument("--file", required=True, type=Path)
        args = parser.parse_args(argv)
        if args.command == "init":
            args.dir.mkdir(parents=True, exist_ok=True)
        elif not args.dir.is_dir():
            raise ResearchError("session_missing", "Initialize the session first")
        with session_lock(args.dir):
            if args.command == "init":
                session = ResearchSession.initialize(args.dir, args.budget, args.seed)
                result = session.status()
            else:
                session = ResearchSession.load(args.dir)
                if args.command == "status":
                    result = session.status()
                elif args.command == "propose":
                    trial = session.propose(json.loads(args.file.read_text()))
                    result = {"trial": explained_trial(trial, session.ontology), "state": session.status()}
                elif args.command == "run":
                    trial = session.run()
                    result = {"trial": explained_trial(trial, session.ontology), "state": session.status()}
                else:
                    result = session.finalize()
        print(json.dumps({"ok": True, "description": COMMAND_DESCRIPTIONS[args.command],
                          "result": result}, allow_nan=False))
        return 0
    except (ResearchError, OSError, ValueError) as exc:
        code = getattr(exc, "code", "io_error" if isinstance(exc, OSError) else "invalid_json")
        print(json.dumps({"ok": False,
                          "description": "The command stopped. Inspect the error and use status to check saved evidence and permitted next steps.",
                          "error": {"code": code, "message": str(exc)}}))
        return 3 if code in ("experiment_failed", "io_error", "invalid_state", "session_busy") else 2


if __name__ == "__main__":
    sys.exit(main())
