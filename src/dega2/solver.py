"""Public DEGA-2 API.

    >>> import dega2
    >>> res = dega2.solve(a, b, c, phi)
    >>> res.x, res.phi
"""
import time
from dataclasses import dataclass

import numpy as np

from . import GA4, GenPop, influ_cof


@dataclass
class Result:
    """Result of a DEGA-2 optimization."""

    x: np.ndarray            # best solution found (vector of length n)
    phi: float               # phi value for this solution
    history: np.ndarray      # minimum phi value at each iteration
    population: np.ndarray   # last iteration's population before selection
    time: float              # computation time in seconds
    iterations: int          # number of iterations performed

    def summary(self, max_vars=None):
        """Return a formatted text summary."""
        x = np.ravel(self.x)
        shown = x if max_vars is None else x[:max_vars]
        apercu = ", ".join(f"{v:.4g}" for v in shown)
        lignes = [
            "DEGA-2 RESULTS",
            f"Best phi      : {self.phi:.6g}",
            f"CPU time      : {self.time:.2f} s",
            f"Iterations    : {self.iterations}",
            f"x*  : {apercu}",
        ]
        largeur = max(len(ligne) for ligne in lignes) + 2
        out = ["┏" + "━" * largeur + "┓"]
        for i, ligne in enumerate(lignes):
            out.append("┃ " + ligne.ljust(largeur - 1) + "┃")
            if i == 0:
                out.append("┣" + "━" * largeur + "┫")
        out.append("┗" + "━" * largeur + "┛")
        return "\n".join(out)

    def plot(self, show=True):
        """Plot the convergence curve (requires matplotlib)."""
        from .plotting import plot_convergence
        return plot_convergence(self, show=show)


def solve(a, b, c, phi, *, n_population=80, alpha=4, explor=3,
          iterations=None, positivity=True, seed=None, verbose=True):
    """Run DEGA-2 on the problem: minimize phi·x subject to a·x <= b, x integer.

    Parameters
    ----------
    a : (m, n) constraint matrix.
    b : (m,) or (m, 1) right-hand side.
    c : (p, n) matrix of p objectives (used for the non-dominance test).
    phi : (n,) or (1, n) scalar objective function to minimize.
    n_population : population size.
    alpha : the phi exploration population has size n_population / alpha.
    explor : number of variables explored per iteration.
    iterations : number of iterations (default 2 * n).
    positivity : if True, add the constraints x >= 0.
    seed : random seed for reproducible results.
    verbose : display the progress bar.

    Returns a `Result` object.
    """
    a = np.atleast_2d(np.asarray(a))
    b = np.asarray(b).reshape(-1, 1)
    c = np.atleast_2d(np.asarray(c))
    phi = np.asarray(phi).reshape(1, -1)

    n = a.shape[1]
    p = c.shape[0]
    if b.shape[0] != a.shape[0]:
        raise ValueError("b must have the same number of rows as a.")
    if c.shape[1] != n or phi.shape[1] != n:
        raise ValueError("a, c, and phi must have the same number of variables n.")
    if not 1 <= explor <= n:
        raise ValueError("explor must be between 1 and n.")
    if n_population < 1 or alpha < 1:
        raise ValueError("n_population and alpha must be >= 1.")

    T = 2 * n if iterations is None else int(iterations)

    if seed is not None:
        np.random.seed(seed)

    if positivity:
        a = np.concatenate((a, -np.eye(n)), axis=0)
        b = np.concatenate((b, np.zeros((n, 1))), axis=0)

    # Population initiale
    population = GenPop.Gen(a, b, c, n, n_population)

    # Influence des coefficients
    p_cof1, index_p, p_cofphi1, index_phi = influ_cof.selct(c, n, p, phi)

    # Algorithme
    t0 = time.perf_counter()
    pop_final, _, min_phi, x_opt = GA4.NewPop(
        population, n_population, explor,
        p_cof1, index_p, p_cofphi1, index_phi,
        phi, alpha, a, b, c, T, n, verbose=verbose)
    elapsed = time.perf_counter() - t0

    history = np.asarray(min_phi, dtype=float)
    return Result(
        x=np.ravel(x_opt),
        phi=float(history[-1]),
        history=history,
        population=pop_final,
        time=elapsed,
        iterations=T,
    )
