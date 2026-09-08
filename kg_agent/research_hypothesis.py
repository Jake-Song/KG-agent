"""Deterministic, evidence-guided proposal generation for research sessions."""

from __future__ import annotations

from .research_domain import ResearchError, ResearchOntology


def _penalty_slug(penalty: int | float) -> str:
    return str(penalty).replace(".", "p")


def _configuration_id(degree: int, penalty: int | float) -> str:
    return f"degree-{degree}-penalty-{_penalty_slug(penalty)}"


def _next_id(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    suffix = 2
    while f"{base}-{suffix}" in used:
        suffix += 1
    return f"{base}-{suffix}"


def generate_hypothesis(state: dict, ontology: ResearchOntology) -> dict:
    """Generate one unseen proposal from the current validation winner.

    This function is deliberately pure: it does not fit a model, evaluate data,
    queue a trial, or mutate ``state``.
    """
    successful = [trial for trial in state["trials"] if trial["status"] == "complete"]
    if not successful:
        raise ResearchError("baseline_required", "A successful baseline is required")
    best = min(successful, key=lambda trial: trial["validation_mse"])
    best_proposal = best["proposal"]
    degree_index = {value: index for index, value in enumerate(ontology.degrees)}
    penalty_index = {value: index for index, value in enumerate(ontology.penalties)}
    best_degree = degree_index[best_proposal["degree"]]
    best_penalty = penalty_index[best_proposal["penalty"]]
    used_configurations = {
        (trial["proposal"]["degree"], trial["proposal"]["penalty"])
        for trial in state["trials"]
    }
    used_ids = {trial["proposal"]["id"] for trial in state["trials"]}

    candidates = []
    for degree in ontology.degrees:
        for penalty in ontology.penalties:
            if (degree, penalty) in used_configurations:
                continue
            degree_delta = degree_index[degree] - best_degree
            penalty_delta = penalty_index[penalty] - best_penalty
            changed_axes = int(degree_delta != 0) + int(penalty_delta != 0)
            distance = abs(degree_delta) + abs(penalty_delta)
            if degree_delta and not penalty_delta:
                axis_priority = 0
                direction_priority = 0 if degree_delta > 0 else 1
            elif penalty_delta and not degree_delta:
                axis_priority = 1
                direction_priority = 0 if penalty_delta > 0 else 1
            else:
                axis_priority = 2
                direction_priority = 0
            candidates.append((
                (changed_axes, distance, axis_priority, direction_priority,
                 degree_index[degree], penalty_index[penalty]),
                degree,
                penalty,
            ))
    if not candidates:
        raise ResearchError("hypothesis_space_exhausted", "No unseen permitted configuration remains")

    _, degree, penalty = min(candidates)
    ident = _next_id(_configuration_id(degree, penalty), used_ids)
    best_id = best_proposal["id"]
    score = best["validation_mse"]
    if degree != best_proposal["degree"] and penalty == best_proposal["penalty"]:
        rationale = (
            f"{best_id} is the current validation winner at MSE {score:.8g}. "
            f"Test {'higher' if degree > best_proposal['degree'] else 'lower'} polynomial capacity "
            f"while holding ridge penalty at {penalty:g}."
        )
    elif penalty != best_proposal["penalty"] and degree == best_proposal["degree"]:
        rationale = (
            f"{best_id} is the current validation winner at MSE {score:.8g}. "
            f"Test {'stronger' if penalty > best_proposal['penalty'] else 'weaker'} ridge regularization "
            f"while holding polynomial degree at {degree}."
        )
    else:
        rationale = (
            f"{best_id} is the current validation winner at MSE {score:.8g}. "
            "All nearer one-factor configurations have been proposed; test the next "
            "unseen configuration in the deterministic local search."
        )
    return {
        "id": ident,
        "hypothesis": (
            f"A degree-{degree} polynomial with ridge penalty {penalty:g} will reduce "
            f"validation MSE by at least 1% versus {best_id}."
        ),
        "rationale": rationale,
        "degree": degree,
        "penalty": penalty,
    }
