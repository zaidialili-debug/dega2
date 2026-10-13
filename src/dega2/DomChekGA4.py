import numpy as np


def _indices_non_domines(obj):
    """Match `non_dominance` without repeated np.delete calls.

    Compare i with all remaining j values in one operation and return the
    indices of retained columns in their original order.
    """
    idx = np.arange(obj.shape[1])
    i = 0
    while i < len(idx) - 1:
        oi = obj[:, idx[i]][:, None]
        reste = obj[:, idx[i + 1:]]
        le = (oi <= reste).all(axis=0)      # i weakly dominates j -> remove j
        gt = (oi > reste).all(axis=0)       # j strictly dominates i -> remove i
        if gt.any():
            jg = int(gt.argmax())           # first j that dominates i
            a_supprimer = np.append(np.flatnonzero(le[:jg]) + i + 1, i)
            idx = np.delete(idx, a_supprimer)   # i stays at the same position
        else:
            idx = np.delete(idx, np.flatnonzero(le) + i + 1)
            i += 1
    return idx


def non_dominance(c, pop, tab_ind3):
    obj_val = np.dot(c, pop)
    idx = _indices_non_domines(obj_val)
    return pop[:, idx], [tab_ind3[k] for k in idx], obj_val[:, idx]


def _filtrer_contre(obj_p, idx_p, obj_ref, idx_ref, limite_fixe=None):
    """Compare each individual in population `idx_p` with reference set `idx_ref`.

    Remove dominated reference individuals; remove a population individual as
    soon as a reference individual strictly dominates it. `limite_fixe`
    reproduces the scan bound from the original second loop.
    """
    i = 0
    while i < len(idx_p):
        oi = obj_p[:, idx_p[i]][:, None]
        L = len(idx_ref) if limite_fixe is None else min(limite_fixe, len(idx_ref))
        ref = obj_ref[:, idx_ref[:L]]
        le = (oi <= ref).all(axis=0)
        gt = (oi > ref).all(axis=0)
        if gt.any():
            jg = int(gt.argmax())
            idx_ref = np.delete(idx_ref, np.flatnonzero(le[:jg]))
            idx_p = np.delete(idx_p, i)     # i stays at the same position
        else:
            idx_ref = np.delete(idx_ref, np.flatnonzero(le))
            i += 1
    return idx_p, idx_ref


def nd_update(c, population, pop1, pop1_phi, tab_ind11, tab_ind11_phi):

    # Apply non-dominance filtering to pop1_phi, then pop1.
    pop1_phi, tab_ind11_phi, obj_val_phi = non_dominance(c, pop1_phi, tab_ind11_phi)
    pop1, tab_ind11, obj_val1 = non_dominance(c, pop1, tab_ind11)

    obj_val_p = np.dot(c, population)
    idx_p = np.arange(population.shape[1])

    # Compare population with pop1.
    idx_p, idx1 = _filtrer_contre(obj_val_p, idx_p, obj_val1,
                                  np.arange(pop1.shape[1]))

    # Compare population with pop1_phi. Preserve the original behavior: the scan
    # of pop1_phi is bounded by the population size (size_popkD), not by the size
    # of pop1_phi. If pop1_phi is larger, its last elements are not compared.
    D = len(idx_p)
    idx_p, idx_phi = _filtrer_contre(obj_val_p, idx_p, obj_val_phi,
                                     np.arange(pop1_phi.shape[1]), limite_fixe=D)

    pop2 = np.concatenate((population[:, idx_p], pop1_phi[:, idx_phi], pop1[:, idx1]),
                          axis=1)
    return (pop2,
            [tab_ind11_phi[k] for k in idx_phi],
            [tab_ind11[k] for k in idx1])