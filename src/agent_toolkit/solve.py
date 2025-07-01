from lp_parser import parse_word_problem
from brute_force_lp import brute_force_lp

def solve_lp_from_prompt(prompt: str, step: int = 1):
    """Parses and solves an LP from a natural language problem description."""
    lp = parse_word_problem(prompt)
    best, report = brute_force_lp(lp, step=step)
    return lp, best, report