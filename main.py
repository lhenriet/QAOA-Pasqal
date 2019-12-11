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
        method='Nelder-Mead',options={'disp': False,'maxiter': 200},tol=10**(-2))

        if abs(res2.fun/float(len(H[-1])))>max2:
            if res2.x[0]>0 and res2.x[1]>0 and res2.x[2]>0:
                max2=abs(res2.fun/float(len(H[-1])))
                indice_max2=mm
                soll2=res2.x
            else:
                continue


    return (-max2,soll2)



def Fig_1_paper(H,psi,H_1,H_2,H_diss,indices):

    seuil=0.9
    current=np.ones(3)
    current[0]=np.pi/2.*random.random()
    current[1]=np.pi/2.*random.random()
    current[2]=np.pi/2.*random.random()


    indice_max2=0
    max2=0.
    soll2=(0.,0.,0.)
    res2=minimize(quantum_routines.QAOA_single_run_observable, current, args=(H,psi,H_1,H_2,H_diss,indices),
        method='Nelder-Mead',options={'disp': False,'maxiter': 200},tol=10**(-2))
    settings.Gamma=0
    (H_1,H_2,H_diss)=Graph_and_vector_space.generate_Hamiltonians(H,indices)
    res3=minimize(quantum_routines.QAOA_single_run_observable, current, args=(H,psi,H_1,H_2,H_diss,indices,2,0),
        method='Nelder-Mead',options={'disp': False,'maxiter': 200},tol=10**(-2))
    if abs(res2.fun/float(len(H[-1])))>max2:
        if res2.x[0]>0 and res2.x[1]>0 and res2.x[2]>0:
            max2=abs(res2.fun/res3.fun)
            soll2=res2.x

    return (-max2,soll2)




def main():
    settings.init()
    String="../results/Fig_1.txt"
    N_it=10
    N_it2=10
    for mm in range(N_it):
        print(mm/N_it)
        settings.Gamma=0.1*float(mm+1)/float(N_it)
        a_val=0.
        compteur=0
        for nn in range(N_it2):
            print(float(nn/N_it2))
            Graph_MIS=Graph_and_vector_space.Graph()
            aaa=Graph_MIS.Divide_non_connected_subgraphs()
            for k in aaa:
                if len(k)>10.:
                    subgraph=Graph_MIS.igraph_representation.subgraph(k)
                    (H,indices)=Graph_and_vector_space.generate_Hilbert_space(subgraph)
                    psi=np.zeros(len(H),dtype=complex)
                    psi[0]=1.+0.*1j
                    settings.Gamma=0.1*float(mm+1)/float(N_it)
                    (H_1,H_2,H_diss)=Graph_and_vector_space.generate_Hamiltonians(H,indices)
                    (a,b)=Fig_1_paper(H,psi,H_1,H_2,H_diss,indices)
                    a_val+=a
                    compteur+=1

        f= open(String,"a+")
        u1=a_val/float(compteur)
        Gamm=0.1*float(mm)/float(N_it)
#                u2=b[1]
#                u3=b[2]
        stri=''
        stri = stri +''. join(str(u1))+","+''. join(str(Gamm))+","#+''. join(str(u3))+","
        f.write(stri+"\n")
        f.close()


if __name__ == "__main__":
    ###TESTING MODULE###
    settings.init()
    mat_diag=np.ones(3)
    mat_Rabi=np.zeros((3, 3))
    mat_Rabi[1,0]=mat_Rabi[0,1]=2

    mat_diss=-1j*np.zeros((3, 3))
    mat_diss[0,0]=-1j
    mat_diss[1,1]=-1j
    mat_diss[2,2]=-2j

    sigma_plus=np.zeros((3, 3))
    sigma_plus[2,0]=sigma_plus[0,2]=1
    tge=np.zeros((3, 3))
    tge[0,0]=1
    AA=quantum_routines.get_derivative_density_matrix(mat_diag,mat_Rabi,mat_diss,sigma_plus)
    print(AA(0.,tge))

#    main()
