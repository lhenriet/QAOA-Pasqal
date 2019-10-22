import Graph_and_vector_space
import quantum_routines
import settings

import scipy.integrate
from scipy.optimize import minimize
import scipy
import igraph
from scipy.sparse import csr_matrix
import random,copy,math
import numpy as np



##Setting up your own experiment
def Concentration_parameters(H,psi,H_1,H_2,H_diss,indices):
    theta=np.ones(4)
    seuil=0.9
    N_random_start=30
    initial_condition=[]
    for m in range(N_random_start):
        current=np.ones(3)
        current[0]=np.pi/2.*random.random()
        current[1]=np.pi/2.*random.random()
        current[2]=np.pi/2.*random.random()
        initial_condition.append(current)

    indice_max2=0
    max2=0.
    soll2=(0.,0.,0.)
    for mm in range(len(initial_condition)):

        res2=minimize(quantum_routines.QAOA_single_run_observable, initial_condition[mm], args=(H,psi,H_1,H_2,H_diss,indices),
        method='Nelder-Mead',options={'disp': False,'maxiter': 200},tol=10**(-4))
    
        if abs(res2.fun/float(len(H[-1])))>max2:
            if res2.x[0]>0 and res2.x[1]>0 and res2.x[2]>0:
                max2=abs(res2.fun/float(len(H[-1])))
                indice_max2=mm
                soll2=res2.x
            else:
                continue


    return (-max2,soll2)





def main():
    settings.init()
    String="../results/Concentration.txt"
    N_it=1000
    for mm in range(N_it):
        print(mm/N_it)
        Graph_MIS=Graph_and_vector_space.Graph()
        aaa=Graph_MIS.Divide_non_connected_subgraphs()
        for k in aaa:
            if len(k)>12.:
                subgraph=Graph_MIS.igraph_representation.subgraph(k)
                (H,indices)=Graph_and_vector_space.generate_Hilbert_space(subgraph)
                psi=np.zeros(len(H),dtype=complex)
                psi[0]=1.+0.*1j
                (H_1,H_2,H_diss)=Graph_and_vector_space.generate_Hamiltonians(H,indices)
                (a,b)=Concentration_parameters(H,psi,H_1,H_2,H_diss,indices)
                f= open(String,"a+")
                u1=b[0]
                u2=b[1]
                u3=b[2]
                stri=''
                stri = stri +''. join(str(u1))+","+''. join(str(u2))+","+''. join(str(u3))+","
                f.write(stri+"\n")
                f.close()


if __name__ == "__main__":
    main()
