"""Scientific vocabulary and executable contracts for the bounded research domain."""
from dataclasses import asdict, dataclass
import math
import re

from .schema import Ontology, RelationSpec


class ResearchError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ActionContract:
    node_type: str
    prerequisite_states: tuple[str, ...]
    evidence: tuple[str, ...]
    completion_state: str = "complete"


class ResearchOntology(Ontology):
    version = 1
    degrees = (1, 2, 3, 4, 5)
    penalties = (0, 0.001, 0.01, 0.1, 1)
    improvement_threshold = 0.01
    actions = {
        "fit": ActionContract("Experiment", ("pending",), ("coefficients",)),
        "evaluate": ActionContract("Evaluation", ("pending",), ("validation_mse",)),
    }

    def __init__(self):
        specs = [
            RelationSpec("contains", domain="Study", range="Experiment"),
            RelationSpec("tests", domain="Experiment", range="Hypothesis"),
            RelationSpec("uses_dataset", domain="Experiment", range="Dataset"),
            RelationSpec("produced", domain="Experiment", range="Model", functional=True),
            RelationSpec("evaluates", domain="Evaluation", range="Model", functional=True),
            RelationSpec("assesses", domain="Evaluation", range="Hypothesis"),
            RelationSpec("compared_with", domain="Hypothesis", range="Experiment"),
            RelationSpec("requires", domain="Evaluation", range="Experiment", dependency=True),
        ]
        super().__init__(specs, satisfied_states=("complete",))

    @property
    def action_map(self):
        return {spec.node_type: name for name, spec in self.actions.items()}

    def describe(self):
        return {
            "version": self.version,
            "proposal_fields": ["id", "hypothesis", "rationale", "degree", "penalty"],
            "degrees": self.degrees, "penalties": self.penalties,
            "improvement_threshold": self.improvement_threshold,
            "actions": {name: asdict(spec) for name, spec in self.actions.items()},
            "rules": ["successful baseline before proposals", "one pending trial",
                      "unique configurations", "fixed trial budget", "finalized study is frozen"],
        }

    def validate_proposal(self, proposal):
        fields = {"id", "hypothesis", "rationale", "degree", "penalty"}
        if not isinstance(proposal, dict) or set(proposal) != fields:
            raise ResearchError("invalid_proposal", f"Expected exactly these fields: {sorted(fields)}")
        if not isinstance(proposal["id"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", proposal["id"]):
            raise ResearchError("invalid_proposal", "ID must be 1-64 letters, digits, underscores or hyphens")
        for name in ("hypothesis", "rationale"):
            if not isinstance(proposal[name], str) or not proposal[name].strip():
                raise ResearchError("invalid_proposal", f"{name} must be nonempty text")
        if type(proposal["degree"]) is not int or proposal["degree"] not in self.degrees:
            raise ResearchError("invalid_proposal", "Degree must be an integer from 1 to 5")
        if type(proposal["penalty"]) not in (int, float) or proposal["penalty"] not in self.penalties:
            raise ResearchError("invalid_proposal", "Penalty is not in the allowed set")

    def check_command(self, command, state):
        trials = state["trials"]
        pending = any(t["status"] == "pending" for t in trials)
        baseline_ok = trials[0]["status"] == "complete"
        if state["status"] == "finalized":
            raise ResearchError("finalized", "The study is frozen")
        if command == "run":
            if not pending:
                raise ResearchError("no_pending_trial", "There is no pending trial")
            return
        if not baseline_ok:
            raise ResearchError("baseline_required", "A successful baseline is required")
        if pending:
            raise ResearchError("pending_trial", "Run the pending trial first")
        if command == "propose" and len(trials) >= state["budget"]:
            raise ResearchError("budget_exhausted", "The trial budget is exhausted")

    def check_action(self, action, node):
        spec = self.actions[action]
        if node.type != spec.node_type or node.attrs.get("status") not in spec.prerequisite_states:
            raise ResearchError("invalid_transition", f"{action} cannot execute on {node.id}")

    def check_evidence(self, action, evidence, degree):
        spec = self.actions[action]
        if set(evidence) != set(spec.evidence):
            raise ResearchError("invalid_evidence", f"Missing or unexpected evidence for {action}")
        values = evidence.get("coefficients", [evidence.get("validation_mse")])
        if (not isinstance(values, list) or not values
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ResearchError("invalid_evidence", "Evidence must contain finite numeric values")
        if action == "fit" and len(values) != degree + 1:
            raise ResearchError("invalid_evidence", "Incorrect coefficient count")
        if action == "evaluate" and values[0] < 0:
            raise ResearchError("invalid_evidence", "MSE cannot be negative")
