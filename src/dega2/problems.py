import numpy as np

def Genr(n,m1,p):
    a=np.random.randint(1,10,(m1,n),dtype=int)

    

    b=np.random.randint(8*n,10*n,m1,dtype=int)[:,np.newaxis]
    

    c=np.random.randint(-10,10,(p,n),dtype=int)
    phi=np.random.randint(-10,10,n,dtype=int)[np.newaxis,:]
    return a,b,c,phi
    

