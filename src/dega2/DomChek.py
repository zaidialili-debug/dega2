import numpy as np


def non_dominance(c, pop):
    obj_val = np.dot(c, pop)
    idx = np.arange(obj_val.shape[1])
    i = 0
    while i < len(idx) - 1:
        oi = obj_val[:, idx[i]][:, None]
        reste = obj_val[:, idx[i + 1:]]
        le = (oi <= reste).all(axis=0)      # i weakly dominates j -> remove j
        gt = (oi > reste).all(axis=0)       # j strictly dominates i -> remove i
        if gt.any():
            jg = int(gt.argmax())           # first j that dominates i
            idx = np.delete(idx, np.append(np.flatnonzero(le[:jg]) + i + 1, i))
        else:
            idx = np.delete(idx, np.flatnonzero(le) + i + 1)
            i += 1
    return pop[:, idx]