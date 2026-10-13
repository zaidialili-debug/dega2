"""Problem and parameter input helpers.

Three ways to provide a problem:
  - a .json or .npz file               -> charger_probleme(path)
  - keyboard input                     -> saisir_probleme()
  - NumPy arrays in your code          -> dega2.solve(a, b, c, phi)
"""
import json

import numpy as np

CLES_PROBLEME = ("a", "b", "c", "phi")
CLES_PARAMETRES = ("n_population", "alpha", "explor", "iterations", "positivity")

PARAMETRES_DEFAUT = {
    "n_population": 80,
    "alpha": 4,
    "explor": 3,
    "iterations": None,      # None -> 2 * n
    "positivity": True,
}


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
def charger_probleme(chemin):
    """Load a problem from a .json or .npz file.

    The file must contain keys a, b, c, phi, and may optionally contain
    "parameters" (n_population, alpha, explor, iterations, positivity).
    The legacy "parametres" key is also accepted.
    Returns (a, b, c, phi, parameters); parameters may be an empty dict.
    """
    chemin = str(chemin)
    bas = chemin.lower()
    if bas.endswith(".json"):
        with open(chemin, encoding="utf-8") as f:
            donnees = json.load(f)
    elif bas.endswith(".npz"):
        with np.load(chemin, allow_pickle=False) as f:
            donnees = {k: f[k] for k in f.files}
    else:
        raise ValueError("Unsupported format: use a .json or .npz file.")

    manquantes = [k for k in CLES_PROBLEME if k not in donnees]
    if manquantes:
        raise ValueError("Missing keys in file: " + ", ".join(manquantes))

    parametres = donnees.get("parameters", donnees.get("parametres", {}))
    if not isinstance(parametres, dict):
        raise ValueError('"parameters" must be a dictionary.')
    inconnus = [k for k in parametres if k not in CLES_PARAMETRES]
    if inconnus:
        raise ValueError("Unknown parameters: " + ", ".join(inconnus)
                         + " (expected: " + ", ".join(CLES_PARAMETRES) + ")")

    a, b, c, phi = (np.asarray(donnees[k]) for k in CLES_PROBLEME)
    return a, b, c, phi, dict(parametres)


def ecrire_exemple(chemin):
    """Write an example .json file that the user can edit."""
    exemple = {
        "a": [[2, 3, 1, 4, 2, 1, 3, 2],
              [1, 1, 2, 3, 1, 4, 2, 1],
              [3, 1, 1, 2, 4, 1, 1, 3]],
        "b": [40, 36, 42],
        "c": [[3, -1, 2, 0, 4, -2, 1, 1],
              [-2, 4, 1, -1, 0, 3, -3, 2],
              [1, 2, -4, 3, -1, 1, 2, -2],
              [0, -3, 2, 2, 1, -1, 4, -1],
              [2, 1, 1, -3, 2, 2, -2, 3]],
        "phi": [-3, -2, -4, -1, -2, -3, -1, -2],
        "parameters": {
            "n_population": 20,
            "alpha": 4,
            "explor": 3,
            "iterations": 16,
            "positivity": True,
        },
    }
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(exemple, f, indent=2, ensure_ascii=False)
    return exemple


# ----------------------------------------------------------------------------
# Interactive input
# ----------------------------------------------------------------------------
def _nombre(brut):
    """Convert text to int if possible, otherwise to float."""
    try:
        return int(brut)
    except ValueError:
        return float(brut)


def demander(invite, defaut=None, type_=int, minimum=None, maximum=None):
    """Prompt until a valid response is provided. An empty response uses the default."""
    while True:
        suffixe = "" if defaut is None else f" [{defaut}]"
        brut = input(f"{invite}{suffixe}: ").strip()
        if brut == "":
            if defaut is not None:
                return defaut
            print("  A value is required.")
            continue
        try:
            valeur = type_(brut)
        except ValueError:
            print("  Invalid value.")
            continue
        if (minimum is not None and valeur < minimum) or \
           (maximum is not None and valeur > maximum):
            borne = f"between {minimum} and {maximum}" if None not in (minimum, maximum) \
                else (f">= {minimum}" if minimum is not None else f"<= {maximum}")
            print(f"  The value must be {borne}.")
            continue
        return valeur


def demander_oui_non(invite, defaut=True):
    indice = "Y/n" if defaut else "y/N"
    while True:
        brut = input(f"{invite} [{indice}]: ").strip().lower()
        if brut == "":
            return defaut
        if brut in ("y", "yes"):
            return True
        if brut in ("n", "no"):
            return False
        print("  Please answer y or n.")


def demander_vecteur(invite, longueur):
    """Read `length` numbers separated by spaces or commas."""
    while True:
        brut = input(f"{invite}: ").replace(",", " ").split()
        try:
            valeurs = [_nombre(t) for t in brut]
        except ValueError:
            print("  Please enter numbers only.")
            continue
        if len(valeurs) != longueur:
            print(f"  Expected {longueur} numbers, got {len(valeurs)}.")
            continue
        return valeurs


def saisir_probleme():
    """Prompt for a, b, c, and phi. Returns (a, b, c, phi)."""
    n = demander("Number of variables n", minimum=1)
    m = demander("Number of constraints m", minimum=1)
    p = demander("Number of objectives p (rows of c)", minimum=1)

    print(f"\nMatrix a: {m} rows of {n} numbers (constraints a·x <= b)")
    a = [demander_vecteur(f"  a, row {i + 1}/{m}", n) for i in range(m)]
    b = demander_vecteur(f"Right-hand side b: {m} numbers", m)
    print(f"\nMatrix c: {p} rows of {n} numbers (objectives)")
    c = [demander_vecteur(f"  c, row {i + 1}/{p}", n) for i in range(p)]
    phi = demander_vecteur(f"Function to minimize phi: {n} numbers", n)
    return np.array(a), np.array(b), np.array(c), np.array(phi)


def saisir_parametres(n, defauts=None):
    """Prompt for algorithm parameters and return a dict accepted by `solve`.

    The suggested number of iterations is 2 * n, where n is the number of variables.
    """
    d = dict(PARAMETRES_DEFAUT)
    d.update(defauts or {})
    print("\nAlgorithm parameters (press Enter to accept the suggested value)")
    n_population = demander("  Population size", d["n_population"], minimum=1)
    alpha = demander("  alpha", d["alpha"], minimum=1)
    explor = demander("  Variables explored per iteration", min(d["explor"], n),
                      minimum=1, maximum=n)
    iterations = demander("  Number of iterations",
                          2 * n if d["iterations"] is None else d["iterations"], minimum=1)
    positivity = demander_oui_non("  Add the constraints x >= 0?", d["positivity"])
    return {"n_population": n_population, "alpha": alpha, "explor": explor,
            "iterations": iterations, "positivity": positivity}
