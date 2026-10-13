"""Demonstration using a random problem."""
import numpy as np

from . import problems
from .solver import solve


def demo(n=30, p=10, m1=15, n_population=80, alpha=4, explor=3,
         iterations=None, seed=None, plot=True, verbose=True):
    """Generate and solve a random problem, then return the `Result`."""
    if seed is not None:
        np.random.seed(seed)

    a, b, c, phi = problems.Genr(n, m1, p)
    T = 2 * n if iterations is None else iterations

    if verbose:
        print("\n🧬  DEGA-2 — starting optimization "
              f"(n={n}, objectives={p}, constraints={m1}, iterations={T})\n")

    res = solve(a, b, c, phi, n_population=n_population, alpha=alpha,
                explor=explor, iterations=T, verbose=verbose)

    if verbose:
        print("\n" + res.summary(max_vars=n) + "\n")
    if plot:
        res.plot()
    return res
