import numpy as np


def chek(pop, a, b):
    if pop.shape[1] == 0:
        return pop
    return pop[:, (a @ pop <= b).all(axis=0)]
        
       
       
    
    

