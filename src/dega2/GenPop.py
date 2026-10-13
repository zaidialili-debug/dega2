import numpy as np
from . import ConsChek, DomChek

# Maximum consecutive samples without progress before giving up.
_ESSAIS_MAX = 5000


def Gen(a,b,c,n,n_population):
    population=np.random.randint(0,3,(n,n_population),dtype=int);
    pop2=ConsChek.chek(population, a, b)
    pop3=DomChek.non_dominance(c, pop2)
    np1=n_population-len(pop3.T)
    sans_progres=0
    while np1!=0:
        population=np.random.randint(0,3,(n,np1),dtype=int);
        pop2=ConsChek.chek(population, a, b)
        pop3=np.concatenate((pop3,pop2),axis=1);
        pop3=DomChek.non_dominance(c, pop3)
        np1_nouveau=n_population-len(pop3.T)
        # Without this guard, the loop could run indefinitely when too few
        # feasible, pairwise non-dominated solutions exist (few objectives or
        # variables, or very tight constraints).
        sans_progres = 0 if np1_nouveau < np1 else sans_progres + 1
        if sans_progres > _ESSAIS_MAX:
            raise RuntimeError(
                "Could not build a population of %d feasible, mutually non-dominated "
                "solutions (%d found). Reduce the population size, add objectives "
                "(rows of c), or relax the constraints."
                % (n_population, n_population - np1_nouveau))
        np1=np1_nouveau

    return(pop3)
