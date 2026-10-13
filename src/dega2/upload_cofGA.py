import numpy as np
from collections import Counter
def upload(p_cof1,index_p,p_cofphi1,index_phi,tab_ind1,tab_ind1_phi,tab_ind11,tab_ind11_phi, explor,condidat_phi,condidat,n_pop,n_pop_phi):
    x1=Counter(tab_ind11)
    x2=Counter(tab_ind11_phi)
    x3=Counter(tab_ind1)
    x4=Counter(tab_ind1_phi)
    counter_tab11=list(x1.items())
    counter_tab11_phi=list(x2.items())
    counter_tab1=list(x3.items())
    counter_tab1_phi=list(x4.items())
   
    
    l1=len(counter_tab1)
    l2=len(counter_tab11)
    l3=len(counter_tab1_phi)
    l4=len(counter_tab11_phi)
    
    if (l1==explor) and (l2==explor):
        rho=[]
        for i in range (0,explor):
            rho.append(counter_tab11[i][1]/counter_tab1[i][1])
    else:
        rho=[0]*explor
        for ii in range(0,explor):
            jj=0
            while jj<l1:
                if condidat[ii]==counter_tab1[jj][0]:
                    tt=0
                    while tt<l2:
                        if counter_tab1[jj][0]==counter_tab11[tt][0]:
                            rho[ii]=counter_tab11[tt][1]/counter_tab1[jj][1]
                        tt=tt+1
                jj=jj+1
                
                
    if (l3==explor) and (l4==explor):
       rho1=[]
       for i in range (0,explor):
           rho1.append(counter_tab11_phi[i][1]/counter_tab1_phi[i][1])
    else:
        rho1=[0]*explor
        for ii in range(0,explor):
            jj=0
            while jj<l3:
                if condidat_phi[ii]==counter_tab1_phi[jj][0]:
                    tt=0
                    while tt<l4:
                        if counter_tab1_phi[jj][0]==counter_tab11_phi[tt][0]:
                            rho1[ii]=counter_tab11_phi[tt][1]/counter_tab1_phi[jj][1]
                        tt=tt+1
                jj=jj+1           
    rho_p=[]
    rho_phi=[]
    for i in range(0,explor):
        rho_p.append((p_cof1[0,i])*(rho[i])**2)
        p_cof1[0,i]=rho_p[i]
        rho_phi.append((p_cofphi1[0,i])*(rho1[i])**2)
        p_cofphi1[0,i]=rho_phi[i] 
    
    p_cof1= p_cof1[::, p_cof1[0,].argsort()[::-1]]
    p_cofphi1= p_cofphi1[::, p_cofphi1[0,].argsort()[::-1]]
    index_p=(p_cof1[1,:])[np.newaxis,:]
    index_phi=(p_cofphi1[1,:])[np.newaxis,:]
    index_p=index_p.astype(int)
    index_phi=index_phi.astype(int)
    return p_cof1,p_cofphi1, index_p,index_phi
               
    
    
    

