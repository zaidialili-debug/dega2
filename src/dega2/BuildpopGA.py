import numpy as np

from . import ConsChekGA4, DomChekGA4


def explorer(population1, n_population, explor, p_cof1, index_p, p_cofphi1, index_phi,
             n_pop, n_pop_phi):
    """Vectorized exploration phase: add +1 or -1 to each selected variable row."""
    blocs_phi, blocs_p = [], []
    tab_ind1, tab_ind1_ph = [], []
    n_p = n_pop - n_pop_phi          # number of columns modified in the second half

    for j in range(explor):
        a1 = np.array(population1[:, 0:n_pop_phi])
        a2 = np.array(population1[:, n_pop_phi:n_population])

        r_phi = index_phi[0, j]
        r_p = index_p[0, j]
        a1[r_phi, :] += 1 if p_cofphi1[2, j] < 0 else -1
        a2[r_p, :n_p] += 1 if p_cof1[2, j] < 0 else -1

        tab_ind1_ph.extend([r_phi] * n_pop_phi)
        tab_ind1.extend([r_p] * n_p)
        blocs_phi.append(a1)
        blocs_p.append(a2)

    a3 = np.concatenate(blocs_phi, axis=1)   # concatenate once
    a4 = np.concatenate(blocs_p, axis=1)
    return a3, a4, tab_ind1, tab_ind1_ph


def Build_PT(population1, n_population, explor, p_cof1, index_p, p_cofphi1, index_phi,
             phi, alpha, a, b, c, n_pop, n_pop_phi, n):

    ###################### Variables exploration ##############################
    a3, a4, tab_ind1, tab_ind1_ph = explorer(
        population1, n_population, explor, p_cof1, index_p, p_cofphi1, index_phi,
        n_pop, n_pop_phi)

    ############ Check the constraints and dominance of new populations ########
    tab_ind11 = tab_ind1.copy()
    tab_ind11_phi = tab_ind1_ph.copy()

    [pop1, pop1_phi, tab_ind11, tab_ind11_phi] = ConsChekGA4.chek(
        a3, a4, a, b, tab_ind11_phi, tab_ind11)

    [pop2, tab_ind11_phi, tab_ind11] = DomChekGA4.nd_update(
        c, population1, pop1, pop1_phi, tab_ind11, tab_ind11_phi)

    return pop2, tab_ind1, tab_ind11, tab_ind1_ph, tab_ind11_phi
