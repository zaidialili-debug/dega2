import os
import tempfile
from unittest import mock

import numpy as np
import pytest

import dega2
from dega2 import GenPop, cli, problems, upload_cofGA


def _petit_probleme(seed=0, n=8, m1=4, p=3):
    np.random.seed(seed)
    return problems.Genr(n, m1, p)


# --------------------------------------------------------------------------- solve
def test_solution_realisable_et_positive():
    a, b, c, phi = _petit_probleme()
    res = dega2.solve(a, b, c, phi, n_population=20, iterations=5,
                      seed=1, verbose=False)
    x = res.x
    assert x.shape == (a.shape[1],)
    assert np.all(x >= 0)
    assert np.all(a @ x <= b.ravel())


def test_historique_et_phi_coherents():
    a, b, c, phi = _petit_probleme()
    res = dega2.solve(a, b, c, phi, n_population=20, iterations=5,
                      seed=1, verbose=False)
    assert len(res.history) == 5
    assert res.phi == pytest.approx(res.history[-1])
    assert res.iterations == 5


def test_reproductible_avec_seed():
    a, b, c, phi = _petit_probleme()
    r1 = dega2.solve(a, b, c, phi, n_population=20, iterations=5, seed=3, verbose=False)
    r2 = dega2.solve(a, b, c, phi, n_population=20, iterations=5, seed=3, verbose=False)
    assert np.array_equal(r1.x, r2.x)
    assert r1.phi == r2.phi


def test_dimensions_incoherentes():
    a, b, c, phi = _petit_probleme()
    with pytest.raises(ValueError):
        dega2.solve(a, b[:-1], c, phi, verbose=False)
    with pytest.raises(ValueError):
        dega2.solve(a, b, c, phi[:-1], verbose=False)


# ------------------------------------------------- mise à jour des coefficients
def _etat_initial():
    p_cof1 = np.array([[0.5, 0.3, 0.2], [0, 1, 2], [1, 1, 1]], dtype=float)
    p_cofphi1 = np.array([[0.6, 0.3, 0.1], [2, 0, 1], [-1, -1, -1]], dtype=float)
    index_p = np.array([[0, 1, 2]])
    index_phi = np.array([[2, 0, 1]])
    return p_cof1, index_p, p_cofphi1, index_phi


def test_upload_utilise_les_effectifs_et_condidat_phi():
    """Variable 1 (p) : 2 survivants sur 4 -> rho = 0,5 ; variable 2 (phi) : aucun
    survivant -> rho = 0 ; variable 0 (phi) : 2 sur 4 -> rho = 0,5."""
    p_cof1, index_p, p_cofphi1, index_phi = _etat_initial()
    tab_ind1 = [0] * 4 + [1] * 4
    tab_ind11 = [1] * 2                       # variable 0 : 0 survivant
    tab_ind1_phi = [2] * 4 + [0] * 4
    tab_ind11_phi = [0] * 2                   # variable 2 : 0 survivant

    p_cof1, p_cofphi1, index_p, index_phi = upload_cofGA.upload(
        p_cof1, index_p, p_cofphi1, index_phi,
        tab_ind1, tab_ind1_phi, tab_ind11, tab_ind11_phi,
        2, [2, 0], [0, 1], 8, 2)

    # p : [0.5*0, 0.3*0.25, 0.2] -> triés : variables 2, 1, 0
    assert index_p.tolist() == [[2, 1, 0]]
    assert p_cof1[0].tolist() == pytest.approx([0.2, 0.075, 0.0])
    # phi : [0.6*0, 0.3*0.25, 0.1] -> triés : colonnes 2, 1, 0 -> variables 1, 0, 2
    assert index_phi.tolist() == [[1, 0, 2]]
    assert p_cofphi1[0].tolist() == pytest.approx([0.1, 0.075, 0.0])


def test_pas_de_nan_dans_un_run():
    a, b, c, phi = _petit_probleme(n=10, m1=5, p=4)
    with np.errstate(all="raise"):
        res = dega2.solve(a, b, c, phi, n_population=24, iterations=12,
                          seed=5, verbose=False)
    assert np.all(np.isfinite(res.history))


# ------------------------------------------------------------------ fichiers
def test_exemple_json_puis_chargement_et_calcul():
    with tempfile.TemporaryDirectory() as d:
        chemin = os.path.join(d, "ex.json")
        dega2.ecrire_exemple(chemin)
        a, b, c, phi, params = dega2.charger_probleme(chemin)
        assert a.shape == (3, 8) and c.shape == (5, 8) and len(phi) == 8
        res = dega2.solve(a, b, c, phi, seed=0, verbose=False, **params)
        assert np.all(a @ res.x <= b)


def test_fichier_invalide():
    with tempfile.TemporaryDirectory() as d:
        mauvais = os.path.join(d, "x.txt")
        open(mauvais, "w").close()
        with pytest.raises(ValueError):
            dega2.charger_probleme(mauvais)
        incomplet = os.path.join(d, "x.json")
        with open(incomplet, "w") as f:
            f.write('{"a": [[1]], "b": [1]}')
        with pytest.raises(ValueError):
            dega2.charger_probleme(incomplet)


# ------------------------------------------------------------ ligne de commande
def test_cli_fichier():
    with tempfile.TemporaryDirectory() as d:
        chemin = os.path.join(d, "ex.json")
        assert cli.main(["--exemple", chemin]) == 0
        assert cli.main(["-f", chemin, "--seed", "1", "--no-plot"]) == 0


def test_cli_assistant_saisie_clavier(capsys):
    # problème à 3 variables, 1 contrainte, 2 objectifs (petit : population de 5)
    reponses = [
        "2",                    # saisie au clavier
        "3", "1", "2",          # n, m, p
        "1 2 3",                # a
        "30",                   # b
        "1 0 -1", "-1 1 0",     # c
        "-1 -1 -2",             # phi
        "5", "", "", "", "",    # population 5, puis valeurs proposées
        "n",                    # pas de courbe
    ]

    def repondre(invite):
        print(invite, end="")
        return reponses.pop(0)

    with mock.patch("builtins.input", side_effect=repondre):
        assert cli.main([]) == 0
    sortie = capsys.readouterr()
    assert "How would you like to provide the problem?" in sortie.out
    assert "Number of variables n" in sortie.out
    assert "DEGA-2 RESULTS" in sortie.out
    assert "Erreur" not in sortie.out


def test_cli_assistant_ctrl_d():
    with mock.patch("builtins.input", side_effect=EOFError):
        assert cli.main([]) == 1


def test_cli_accepts_english_option_names():
    args = cli._construire_parseur().parse_args([
        "--problem", "problem.json", "--no-positivity", "--explored-variables", "2",
    ])
    assert args.probleme == "problem.json"
    assert args.sans_positivite is True
    assert args.explor == 2


def test_population_impossible_ne_boucle_pas():
    # 1 seule variable binaire-ish, 1 objectif : pas 50 solutions non dominées
    a = np.array([[1]])
    b = np.array([[5]])
    c = np.array([[1]])
    with pytest.raises(RuntimeError):
        GenPop.Gen(a, b, c, 1, 50)
