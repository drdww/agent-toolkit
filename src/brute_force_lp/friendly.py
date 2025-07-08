"""
Friendly brute-force LP solver with live progress prints.

* Works for any number of integer decision variables.
* Supports <=, >=, = constraints and max/min objectives.
* Prints every `print_every` feasible combo so students can see the search.
"""

from __future__ import annotations
import pandas as pd

# ---------------------------------------------------------------------
def _feasible(trial: dict, constraints: list[dict], vars_: list[str]) -> bool:
    """Return True if 'trial' satisfies every constraint."""
    for con in constraints:
        lhs = sum(con["coeff"].get(v, 0) * trial[v] for v in vars_)
        rhs   = con["rhs"]
        sense = con.get("sense", "<=")
        if (sense == "<=" and lhs > rhs) or \
           (sense == ">=" and lhs < rhs) or \
           (sense == "="  and lhs != rhs):
            return False
    return True


# ---------------------------------------------------------------------
def brute_force_lp_friendly(
    lp: dict,
    step: int = 1,
    print_every: int | None = 10_000,
):
    """
    Exhaustive search over integer grid.

    Parameters
    ----------
    lp : dict
        Parsed LP JSON from `parse_word_problem`.
    step : int, default 1
        Grid spacing for each variable.
    print_every : int or None
        Print every Nth feasible combo (None = silent).

    Returns
    -------
    best : dict   – best variable mix + 'objective'
    report : pd.DataFrame – LHS / RHS / slack for each constraint
    """
    vars_   = list(lp["vars"].keys())
    bounds  = [lp["vars"][v]["ub"] for v in vars_]

    obj_sense = lp["objective"].get("sense", "max").lower()
    best_obj  = -float("inf") if obj_sense == "max" else float("inf")
    best_soln = None
    counter   = 0  # counts feasible combos

    # -------------------- recursive nested loops --------------------
    def recurse(level: int, trial: dict):
        nonlocal best_obj, best_soln, counter
        if level == len(vars_):  # all vars assigned
            if not _feasible(trial, lp["constraints"], vars_):
                return
            counter += 1
            if print_every and counter % print_every == 0:
                print(f"[feasible #{counter:>6}] {trial}")
            obj = sum(lp["objective"]["coeff"][v] * trial[v] for v in vars_)
            better = obj > best_obj if obj_sense == "max" else obj < best_obj
            if better:
                best_obj, best_soln = obj, trial.copy()
            return

        var = vars_[level]
        for val in range(0, bounds[level] + 1, step):
            recurse(level + 1, trial | {var: val})

    recurse(0, {})

    if best_soln is None:
        raise ValueError("No feasible solution found.")

    best_soln["objective"] = best_obj

    # -------------------- build constraint report ------------------
    rows = []
    for con in lp["constraints"]:
        lhs = sum(con["coeff"].get(v, 0) * best_soln[v] for v in vars_)
        sense = con.get("sense", "<=")
        slack = (
            con["rhs"] - lhs if sense == "<=" else
            lhs - con["rhs"] if sense == ">=" else 0
        )
        rows.append(
            dict(constraint=con["name"], lhs=lhs, rhs=con["rhs"],
                 sense=sense, slack=slack)
        )
    report = pd.DataFrame(rows)

    print(f"\n🔍 Searched {counter} feasible combos.")
    return best_soln, report
