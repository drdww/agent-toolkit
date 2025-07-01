"""
Brute-force & Monte-Carlo solvers for small integer LPs.

Input = JSON dict that matches the schema produced by lp_parser.parse_word_problem().
"""

from __future__ import annotations
import itertools, numpy as np, pandas as pd

# ---------------------------------------------------------------------
def brute_force_lp(lp: dict, step: int = 1):
    """
    Exhaustively enumerate every integer combination up to each var's 'ub',
    stepping by `step` (default 1).  Returns:
      best_dict  – dict of variable values + 'objective'
      report_df  – dataframe: constraint | lhs | rhs
    """
    vars_   = list(lp["vars"].keys())
    bounds  = [lp["vars"][v]["ub"] for v in vars_]
    best    = {"objective": -float("inf")}

    # Cartesian product of all integer counts
    for combo in itertools.product(*(range(0, ub + 1, step) for ub in bounds)):
        trial = dict(zip(vars_, combo))

        # ---- Feasibility check -------------------------------------------
        feasible = True
        for con in lp["constraints"]:
            lhs = sum(con["coeff"][v] * trial[v] for v in vars_)
            if lhs > con["rhs"]:
                feasible = False
                break
        if not feasible:
            continue

        # ---- Objective ----------------------------------------------------
        obj = sum(lp["objective"]["coeff"][v] * trial[v] for v in vars_)
        if obj > best["objective"]:
            best = trial | {"objective": obj}

    # ---- Build LHS/RHS report --------------------------------------------
    rows = []
    for con in lp["constraints"]:
        lhs = sum(con["coeff"][v] * best[v] for v in vars_)
        rows.append({"constraint": con["name"], "lhs": lhs, "rhs": con["rhs"]})
    report = pd.DataFrame(rows)

    return best, report


# ---------------------------------------------------------------------
def sample_lp(lp: dict, num_samples: int = 20_000, seed: int | None = None):
    """
    Monte-Carlo sampler for quicker 'good-enough' solutions.
    Returns top 10 feasible samples sorted by objective.
    """
    rng    = np.random.default_rng(seed)
    vars_  = list(lp["vars"].keys())
    bounds = [lp["vars"][v]["ub"] for v in vars_]
    rows   = []

    for _ in range(num_samples):
        trial = {v: rng.integers(0, ub + 1) for v, ub in zip(vars_, bounds)}

        # Feasibility
        if any(
            sum(con["coeff"][v] * trial[v] for v in vars_) > con["rhs"]
            for con in lp["constraints"]
        ):
            continue

        obj = sum(lp["objective"]["coeff"][v] * trial[v] for v in vars_)
        rows.append(trial | {"objective": obj})

    return (
        pd.DataFrame(rows)
        .sort_values("objective", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )