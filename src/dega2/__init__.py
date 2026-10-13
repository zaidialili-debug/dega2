"""DEGA-2: a multi-objective genetic algorithm for integer variables."""

__version__ = "0.1.0"

from .solver import Result, solve
from .entree import charger_probleme, ecrire_exemple
from .example import demo

load_problem = charger_probleme
write_example = ecrire_exemple

__all__ = [
    "solve",
    "load_problem",
    "write_example",
    "charger_probleme",
    "ecrire_exemple",
    "demo",
    "Result",
    "__version__",
]
