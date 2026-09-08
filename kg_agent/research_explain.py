"""Human-readable explanations derived from saved research evidence.

These describe the workflow and current evidence; they are not execution logs.
"""

COMMAND_DESCRIPTIONS = {
    "init": "Freeze the seed and trial budget, create the study graph, and queue the linear baseline.",
    "status": "Read saved evidence and check the ontology rules to identify permitted next commands.",
    "propose": "Validate the configuration and study rules, freeze the comparison target, and queue the hypothesis. An identical ID returns its existing trial.",
    "run": "Use graph dependencies to fit the pending model before evaluating it, check evidence contracts, and save the measured conclusion.",
    "finalize": "Select the lowest validation MSE, evaluate its saved model on the test split, and freeze the study. If already finalized, return saved scores without reevaluation.",
}


def trial_steps(trial, ontology):
    p = trial["proposal"]
    complete = trial["status"] == "complete"
    failed = trial["status"] == "failed"
    fitted = "coefficients" in trial
    evaluated = "validation_mse" in trial
    target = trial["comparison_id"]
    threshold = ontology.improvement_threshold * 100
    steps = [
        {"step": "hypothesis", "status": "recorded",
         "description": f"Question: {p['hypothesis']} Reason: {p['rationale']}"},
        {"step": "contract", "status": "accepted",
         "description": (f"Degree {p['degree']} and ridge penalty {p['penalty']} define this trial. "
                         "The baseline is supplied by the project; subsequent proposals must pass "
                         "configuration, uniqueness, baseline, pending-trial, and budget checks.")},
        {"step": "plan", "status": "complete" if complete else "failed" if failed else "pending",
         "description": "The evaluation requires its experiment in the graph. The planner therefore orders fit before evaluate."},
        {"step": "fit", "status": "complete" if fitted else "failed" if failed else "pending",
         "description": (f"Fit degree-{p['degree']} polynomial coefficients using 80 training observations "
                         f"and ridge penalty {p['penalty']}; the intercept is unpenalized. "
                         "The contract requires the correct number of finite coefficients.")},
        {"step": "evaluate", "status": "complete" if evaluated else "failed" if failed and fitted else "skipped" if failed else "pending",
         "description": "Measure mean squared prediction error on 40 fixed validation observations. Lower is better; the contract rejects negative or non-finite scores."},
    ]
    if fitted:
        steps[3]["description"] += f" Saved coefficients: {trial['coefficients']}."
    if evaluated:
        steps[4]["description"] += f" Saved validation MSE: {trial['validation_mse']:.8g}."
    description = (f"Compare with the best trial at submission time ({target}). "
                   f"The hypothesis meets its criterion only if validation MSE decreases by at least {threshold:g}%."
                   if target else "Establish the baseline score as the reference for subsequent hypotheses.")
    if complete and target:
        improvement = trial["relative_improvement"] * 100
        description += (f" Measured relative change: {improvement:+.4f}% improvement. "
                        f"Criterion {'met' if trial['conclusion'] == 'observed_improvement' else 'not met'}. "
                        "A smaller gain can still become the lowest-MSE model without meeting this criterion.")
    elif complete:
        description += " Baseline established; no improvement claim is made."
    if failed:
        description += f" Trial failed: {trial['error']['message']}. No completed hypothesis comparison is available."
    steps.extend([
        {"step": "compare", "status": "complete" if complete else "skipped" if failed else "pending",
         "description": description},
        {"step": "persist", "status": "saved",
         "description": (f"The saved trial is {trial['status']}. Its proposal, evidence, and graph state "
                         "survive a process restart. This trial occupies one budget slot.")},
    ])
    return steps


def explained_trial(trial, ontology):
    return {**trial, "steps": trial_steps(trial, ontology)}


def next_step_description(state):
    if state["status"] == "finalized":
        return "The study is finalized. Read the report; finalize can regenerate it from saved scores. No more trials are permitted."
    if any(t["status"] == "pending" for t in state["trials"]):
        return "Run the pending trial. Its validation evidence is needed before proposing another experiment or finalizing."
    if state["trials"][0]["status"] != "complete":
        return "The baseline failed, so the study is blocked. Inspect the recorded failure; proposals and finalization are not permitted."
    if len(state["trials"]) >= state["budget"]:
        return "The budget is exhausted. Finalize the validation winner to obtain the final test score."
    return "Review the accumulated validation evidence, then propose a justified configuration or finalize if no useful hypothesis remains."
