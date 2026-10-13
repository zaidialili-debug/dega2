import numpy as np


def _realisables(a, b, pop):
    """Return columns of `pop` satisfying a @ x <= b.

    All columns are checked in a single matrix multiplication.
    """
    if pop.shape[1] == 0:
        return np.arange(0)
    return np.flatnonzero((a @ pop <= b).all(axis=0))


def chek(pop_phi, pop, a, b, tab_ind11_phi, tab_ind11):
    idx = _realisables(a, b, pop)
    idx_phi = _realisables(a, b, pop_phi)

    return (pop[:, idx],
            pop_phi[:, idx_phi],
            [tab_ind11[k] for k in idx],
            [tab_ind11_phi[k] for k in idx_phi])