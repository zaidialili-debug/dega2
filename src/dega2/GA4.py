import numpy as np
from . import BuildpopGA, GenPop, upload_cofGA
from tqdm import tqdm


def NewPop(population1, n_population, explor, p_cof1, index_p, p_cofphi1, index_phi,
           phi, alpha, a, b, c, T, n, verbose=True):

    n_pop = len(population1.T)
    n_pop_phi = int(n_pop / alpha)

    min_phi = []

    barre = tqdm(
        range(T),
        desc="DEGA-2",
        unit="iteration",
        colour="green",
        disable=not verbose,
        bar_format="{desc} {percentage:3.0f}%|{bar:35}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for _ in barre:
        [pop2, tab_ind1, tab_ind11, tab_ind1_phi, tab_ind11_phi] = BuildpopGA.Build_PT(
            population1, n_population, explor, p_cof1, index_p, p_cofphi1, index_phi,
            phi, alpha, a, b, c, n_pop, n_pop_phi, n)

        s = pop2.shape[1]

        if s > n_population:
            # Select the n_pop best individuals (ascending phi order).
            xphi1 = np.dot(phi, pop2)
            ind_phi1 = np.argsort(xphi1, axis=1)
            pop3 = pop2[:, ind_phi1[0, :n_pop]].astype(int)
        elif s < n_population:
            pop3 = GenPop.Gen(a, b, c, n, n_population - s)
            pop3 = np.concatenate((pop2, pop3), axis=1)
        else:
            pop3 = pop2

        population1 = pop3
        xphi1 = np.dot(phi, pop3)

        xphi2 = xphi1.min()
        min_phi.append(xphi2)

        ##################### Candidats ####################################
        condidat_phi = list(index_phi[0, :explor])
        condidat = list(index_p[0, :explor])

        ##################### Update coefficients ##########################
        [p_cof1, p_cofphi1, index_p, index_phi] = upload_cofGA.upload(
            p_cof1, index_p, p_cofphi1, index_phi,
            tab_ind1, tab_ind1_phi, tab_ind11, tab_ind11_phi,
            explor, condidat_phi, condidat, n_pop, n_pop_phi)

        barre.set_postfix_str(f"min_phi={xphi2:.4e}", refresh=False)

    # The index comes from pop3 (= population1), not pop2.
    min_index = xphi1.argmin()
    xopt = population1[:, min_index]

    return pop2, xphi2, min_phi, xopt