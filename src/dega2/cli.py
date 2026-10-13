"""Command-line interface:

    dega2                        interactive assistant
    dega2 -f problem.json        problem and parameters loaded from a file
    dega2 --example template.json writes an example file to edit
    dega2 --demo                 random demonstration problem
"""
import argparse

import numpy as np

from . import __version__, problems
from .entree import (PARAMETRES_DEFAUT, charger_probleme, demander,
                     demander_oui_non, ecrire_exemple, saisir_parametres,
                     saisir_probleme)
from .example import demo
from .solver import solve


def _construire_parseur():
    p = argparse.ArgumentParser(
        prog="dega2",
        description="DEGA-2. With no options, start an assistant that prompts for "
                    "the problem and algorithm parameters.")
    p.add_argument("-f", "--problem", "--probleme", dest="probleme", metavar="FILE",
                   help="a .json or .npz file containing a, b, c, phi, and optionally "
                        "\"parameters\" (legacy \"parametres\" is also supported)")
    p.add_argument("--example", "--exemple", dest="exemple", metavar="FILE",
                   help="write an example .json file, then exit")
    p.add_argument("--demo", action="store_true",
                   help="random demonstration problem (see -n, -p, -m)")
    p.add_argument("-n", type=int, default=None, help="[--demo] number of variables (default 30)")
    p.add_argument("-p", "--objectives", "--objectifs", dest="objectifs",
                   metavar="OBJECTIVES", type=int, default=None,
                   help="[--demo] number of objectives (default 10)")
    p.add_argument("-m", "--constraints", "--contraintes", dest="contraintes",
                   metavar="CONSTRAINTS", type=int, default=None,
                   help="[--demo] number of constraints (default 15)")
    p.add_argument("-T", "--iterations", type=int, default=None,
                   help="number of iterations (default 2n)")
    p.add_argument("--population", type=int, default=None, help="population size (default 80)")
    p.add_argument("--alpha", type=int, default=None, help="alpha parameter (default 4)")
    p.add_argument("--explored-variables", "--explor", dest="explor",
                   metavar="COUNT", type=int, default=None,
                   help="variables explored per iteration (default 3)")
    p.add_argument("--no-positivity", "--sans-positivite", dest="sans_positivite",
                   action="store_true",
                   help="do not add the constraints x >= 0")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--no-plot", action="store_true", help="do not display the convergence plot")
    p.add_argument("--version", action="version", version=f"dega2 {__version__}")
    return p


def _lancer(a, b, c, phi, parametres, seed=None, tracer=True):
    a, c = np.atleast_2d(a), np.atleast_2d(c)
    print(f"\n🧬  DEGA-2 — starting (variables={a.shape[1]}, objectives={c.shape[0]}, "
          f"constraints={a.shape[0]})\n")
    res = solve(a, b, c, phi, seed=seed, **parametres)
    print("\n" + res.summary() + "\n")
    if tracer:
        try:
            res.plot()
        except ImportError as exc:
            print(exc)
    return res


def _assistant():
    print("\n🧬  DEGA-2 — assistant\n")
    print("How would you like to provide the problem?")
    print("  1) Load a file (.json or .npz)")
    print("  2) Enter it at the keyboard")
    print("  3) Random problem (demonstration)")
    choix = demander("Choice", 1, minimum=1, maximum=3)

    defauts = {}
    if choix == 1:
        while True:
            chemin = input("File path: ").strip().strip('"').strip("'")
            try:
                a, b, c, phi, defauts = charger_probleme(chemin)
                break
            except (OSError, ValueError) as exc:
                print(f"  Could not read file: {exc}")
    elif choix == 2:
        a, b, c, phi = saisir_probleme()
    else:
        n = demander("Number of variables n", 30, minimum=1)
        m1 = demander("Number of constraints", 15, minimum=1)
        p = demander("Number of objectives", 10, minimum=1)
        a, b, c, phi = problems.Genr(n, m1, p)

    n = np.atleast_2d(a).shape[1]
    parametres = saisir_parametres(n, defauts)
    tracer = demander_oui_non("Display the convergence plot at the end?", True)
    _lancer(a, b, c, phi, parametres, tracer=tracer)


def main(argv=None):
    args = _construire_parseur().parse_args(argv)

    try:
        if args.exemple:
            ecrire_exemple(args.exemple)
            print(f"Example file written: {args.exemple}\n"
                  f"Edit it, then run:  dega2 --problem {args.exemple}")
            return 0

        if args.demo:
            demo(n=30 if args.n is None else args.n,
                 p=10 if args.objectifs is None else args.objectifs,
                 m1=15 if args.contraintes is None else args.contraintes,
                 n_population=80 if args.population is None else args.population,
                 alpha=4 if args.alpha is None else args.alpha,
                 explor=3 if args.explor is None else args.explor,
                 iterations=args.iterations, seed=args.seed, plot=not args.no_plot)
            return 0

        if args.probleme:
            a, b, c, phi, du_fichier = charger_probleme(args.probleme)
            params = dict(PARAMETRES_DEFAUT)
            params.update(du_fichier)                       # valeurs du fichier
            for cle, valeur in (("n_population", args.population), ("alpha", args.alpha),
                                ("explor", args.explor), ("iterations", args.iterations)):
                if valeur is not None:                      # options de la ligne de commande
                    params[cle] = valeur
            if args.explor is None and "explor" not in du_fichier:
                params["explor"] = min(params["explor"], np.atleast_2d(a).shape[1])
            if args.sans_positivite:
                params["positivity"] = False
            _lancer(a, b, c, phi, params, seed=args.seed, tracer=not args.no_plot)
            return 0

        _assistant()
        return 0

    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted.")
        return 1
