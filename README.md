# DEGA-2

An algorithm  of optimization  a linear function over the efficient set of a
multi-objective problem for integer variables, using a genetic algorithm, by Ali Zaidi.

It minimizes a function `phi·x` subject to linear constraints `a·x <= b`
(`x` integer, and `x >= 0` by default), using `p` objectives `c·x` for the
non-dominance test.

## Installation

```bash
pip install dega2
# with the convergence plot:
pip install "dega2[plot]"
```

## Entering your problem and parameters

### 1. Interactive assistant

```bash
dega2
```

The assistant asks for the problem (file, keyboard entry, or a random example),
then for the parameters (population size, alpha, number of explored variables,
iterations, `x >= 0` constraints). Press Enter to accept the suggested value.

Example of entering a problem at the keyboard:

```text
$ dega2

🧬  DEGA-2 — assistant

How would you like to provide the problem?
  1) Load a file (.json or .npz)
  2) Enter it at the keyboard
  3) Random problem (demonstration)
Choice [1]: 2
Number of variables n: 3
Number of constraints m: 1
Number of objectives p (rows of c): 2

Matrix a: 1 rows of 3 numbers (constraints a·x <= b)
  a, row 1/1: 1 2 3
Right-hand side b: 1 numbers: 30

Matrix c: 2 rows of 3 numbers (objectives)
  c, row 1/2: 1 0 -1
  c, row 2/2: -1 1 0
Function to minimize phi: 3 numbers: -1 -1 -2

Algorithm parameters (press Enter to accept the suggested value)
  Population size [80]: 5
  alpha [4]:
  Variables explored per iteration [3]:
  Number of iterations [6]:
  Add the constraints x >= 0? [Y/n]:
Display the convergence plot at the end? [Y/n]: n
```

Here, the user enters the problem and chooses a population size of 5; pressing Enter
accepts the suggested values for the other algorithm parameters.

### 2. `.json` file

```bash
dega2 --example my_problem.json     # writes a template to edit
dega2 --problem my_problem.json     # runs the computation
```

File format:

```json
{
  "a": [[2, 3, 1], [1, 1, 2]],
  "b": [20, 15],
  "c": [[3, -1, 2], [-2, 4, 1]],
  "phi": [-3, -2, -4],
  "parameters": {
    "n_population": 20, "alpha": 4, "explor": 3,
    "iterations": 10, "positivity": true
  }
}
```

- `a`: constraint matrix (m rows, n columns); `b`: right-hand side (m values)
- `c`: objectives matrix (p rows, n columns); `phi`: function to minimize (n values)
- `parameters` is optional. Legacy files using `parametres` are also supported.
  Command-line options (`--population`, `--alpha`, `--explor`, `-T`,
  `--no-positivity`, `--seed`, `--no-plot`) take precedence over
  the file. A `.npz` file (keys `a`, `b`, `c`, `phi`) is also accepted.

### 3. In Python code

```python
import numpy as np
import dega2

a   = np.array([[2, 3, 1], [1, 1, 2]])
b   = np.array([20, 15])
c   = np.array([[3, -1, 2], [-2, 4, 1]])
phi = np.array([-3, -2, -4])

res = dega2.solve(a, b, c, phi, n_population=20, explor=3, iterations=10, seed=0)

print(res.phi)      # best value of phi
print(res.x)        # best solution
res.plot()          # convergence curve (requires matplotlib)

# or from a file:
a, b, c, phi, params = dega2.load_problem("my_problem.json")
res = dega2.solve(a, b, c, phi, **params)
```

`dega2.load_problem` and `dega2.write_example` are the English API names.
The original `charger_probleme` and `ecrire_exemple` names remain available
for backward compatibility.

`solve` parameters: `n_population=80`, `alpha=4`, `explor=3`,
`iterations=None` (2·n by default), `positivity=True`, `seed=None`, `verbose=True`.
`solve` returns a `Result` object: `x`, `phi`, `history` (minimum of phi at each
iteration), `population`, `time`, `iterations`.

Note: the algorithm builds a population of `n_population` feasible solutions that
are mutually non-dominated. With very few objectives or variables, such a
population may not exist; `dega2` then stops with an error message (reduce
`n_population` or add objectives).

## Random demonstration

```bash
dega2 --demo --seed 1
dega2 --demo -n 20 -p 5 --no-plot
```

## Tests

```bash
pip install -e ".[test]"
pytest
```
