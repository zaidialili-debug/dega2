import numpy as np


def selct(c, n, p, phi):
    # Normalized coefficients (vectorized version of the original nested loops).
    sum_c = np.sum(np.abs(c), axis=1)[:, np.newaxis]
    sum_phi = np.sum(np.abs(phi))
    cof_c = np.sum(c / sum_c, axis=0)       # cumulative influence of each variable
    cof_phi = phi[0, :] / sum_phi

    p_cof1 = np.empty((3, n))
    p_cofphi1 = np.empty((3, n))

    p_cof1[0, :] = np.abs(cof_c)
    p_cofphi1[0, :] = np.abs(cof_phi)
    p_cof1[1, :] = np.arange(n)
    p_cofphi1[1, :] = np.arange(n)
    p_cof1[2, :] = np.where(cof_c < 0, -1, 1)
    # Preserve the original behavior: p_cofphi always has a sign of -1
    # (both branches of the original `if` returned -1). See note.
    p_cofphi1[2, :] = -1

    # Sort by descending influence (same expression as the original).
    p_cof1 = p_cof1[::, p_cof1[0, ].argsort()[::-1]]
    p_cofphi1 = p_cofphi1[::, p_cofphi1[0, ].argsort()[::-1]]
    index_p = (p_cof1[1, :])[np.newaxis, :].astype(int)
    index_phi = (p_cofphi1[1, :])[np.newaxis, :].astype(int)

    return [p_cof1, index_p, p_cofphi1, index_phi]