import numpy as np
import random,copy
from scipy.sparse import csr_matrix
import scipy.integrate
from numpy import linalg
import time
import settings
import math


####PAULI OPERATORS####

def sigma_x_operator(basis_vector,indices,pos_sigma=-1,**kwargs):
    """Operator that creates the matrix representation of sigma_x."""
    M=sigma_plus_operator(basis_vector,indices,pos_sigma=-1,**kwargs)
    return M+np.transpose(M)


def sigma_plus_operator(basis_vector,indices,pos_sigma=-1,**kwargs):
    """
    Operator that creates the matrix representation of sigma_+
        we create an overloading variable pos_sigma, that denotes the position of the sigma_+
        if pos_sigma=-1, global operation.
    """
    dim=len(basis_vector)
    sigma_x_matrix=np.zeros((dim, dim))                             #creation of the output array
    for ii in range(dim-1):                                         #Not optimized. We need to implement a 'for basis_vector_loc in basis_vector'
        basis_vector_ii=basis_vector[ii]
        (n_initial,n_final)=get_indices(basis_vector[ii],indices)
        #we look for possible connections in a restricted set, in order to reduce the computation
        #time. The function get_indices will return the indices between which to look.
        if n_initial<0. or n_final<0.:
            continue
        for jj in range(n_initial,n_final):
            basis_vector_jj=basis_vector[jj]
            if pos_sigma>-0.1:                                                  #Local sigma_x
                loc1=list(copy.copy(basis_vector_ii))
                loc1.append(pos_sigma)                                          #we add the index j to the smallest list
                if set(loc1) == set(basis_vector_jj):
                    sigma_x_matrix[ii,jj]=1.
                    continue
            else:                                                              #Global sigma_x
                if(set(basis_vector_ii).issubset(set(basis_vector_jj))):       #here issubset is sufficient because we know that basis_vector_ii and
                    sigma_x_matrix[ii,jj]=1.                                   #basis_vector_jj only differ by one excitation (thanks to get_indices).
    return sigma_x_matrix




def sigma_z_operator(basis_vector,pos=-1,**kwargs):
    """
    Operator that creates the matrix representation of sigma_z. As sigma^z is diagonal in the computat ional basis,
    we will only return a vector-type array and later apply element-wise multiplication with the wavefunction
    if pos=-1, global operation.
    """
    dim=len(basis_vector)
    sigma_z_matrix=np.zeros(dim)

    #Local operator at position pos
    if pos>-0.1:
        for jj in range(dim):
            if (set([pos]).issubset(set(basis_vector[jj]))):
                sigma_z_matrix[jj]=1.

    #Global operator, all positions
    else:
        for jj in range(dim):
            leng=len(basis_vector[jj])
            sigma_z_matrix[jj]=leng

    return sigma_z_matrix



def sigma_z_z_operator(basis_vector,pos_1,pos_2,**kwargs):
    """
    Operator that creates the matrix representation of sigma_z(pos_1)sigma_z(pos_2).
    As it is diagonal in the computational basis, we will only return a vector-type array and
    later apply element-wise multiplication with the wavefunction.
    """
    dim=len(basis_vector)
    sigma_z_z_matrix=np.zeros(dim)
    for jj in range(dim):
        if (set([pos_1,pos_2]).issubset(set(basis_vector[jj]))):
            sigma_z_z_matrix[jj]=1.
    return sigma_z_z_matrix



def get_indices(basis_vector_loc,indices):
    """
    This function will return the indices for which the basis vectors are possibly
    connected to the input vector by a sigma^x operator. Increasing number of excitations.
    """
    n_initial=indices[len(basis_vector_loc)+1]
    if not len(basis_vector_loc)+2<len(indices):
        return (-1,-1)
    n_final=indices[len(basis_vector_loc)+2]
    return (n_initial,n_final)






###OBSERVABLES ROUTINES

def expectation_value(psi,H_2):
    """Function that computes the expectation value of H_2. """
    Hpsi=np.multiply(H_2,psi)
    return np.vdot(psi,Hpsi)


def expected_shortfall(H,psi,H_2,seuil,**kwargs):
    """Function that computes the expected shortfall of H_2. Measurements in the computational basis."""
    val=0.
    prob=0.
    integer=len(psi)-1
    while prob<(seuil-0.00001):
        prob+=abs(psi[integer])**2
        val+=abs(psi[integer])**2*len(H[integer])
        integer-=1
    return -val/prob


def compute_observable(H,psi,H_2,**kwargs):
    """Function called to procede to observable evaluation."""
    if settings.type_observable[0]=="energy":
        return (expectation_value(psi,H_2)).real
    elif settings.type_observable[0]=="cVAR":
        if settings.type_observable[1]==0.:
            raise ValueError('could not find a positive threshold value for the expected shortfall')
        else:
            progressive=kwargs.get('var_progressive',False)
            if not progressive:
                return (expected_shortfall(H,psi,H_2,settings.type_observable[1])).real
            else:
                seuil_progressive=kwargs.get('seuil_var_progressive',False)
                return (expected_shortfall(H,psi,H_2,seuil_progressive)).real







####TIME-EVOLUTION ROUTINES#####

def get_derivative(mat1,mat2,**kwargs):
    """Decorated. Returns function for t-evolution using scipy.integrate.solve_ivp"""

    tunneling=kwargs.get('tunneling','on')
    if tunneling=='off':
        def H_on_psi_loc(tt,yy):
            return -1j*np.multiply(mat1,yy)
        return H_on_psi_loc
    else:
        def H_on_psi_loc(tt,yy):
            return -1j*np.multiply(mat1,yy)-1j*(mat2 @yy)
        return H_on_psi_loc



def evol_scipy(psi,mat1,mat2,tf,**kwargs):
    """
    Main time-evolution function for the wavefunction.
    """

    dissipative=settings.dissipation
    indices=kwargs.get('indices',0.)
    basis_vector=kwargs.get('basis_vector',0.)
    mat2=csr_matrix(mat2)
    H_on_psi=get_derivative(mat1,mat2,**kwargs)
    t_span=(0.,tf)
    #Coherent time-evolution.
    if not dissipative:
        sol=scipy.integrate.solve_ivp(H_on_psi, t_span, psi, method='RK45',
                                        t_eval=None, dense_output=False,
                                        events=None, vectorized=False)
        values=sol.y
        psi=values[ : , -1]

    #Dissipative time-evolution. Jumps allowed.
    else:
        rn=random.random()                                          #random number. if norm(psi)<rn, then jump.
        is_norm_positive=get_test_jump(rn)                          #Automatic stopping of the time-evolution
        finished=False                                              #if the norm of psi gets below rn.
        while not finished:
            sol=scipy.integrate.solve_ivp(H_on_psi, t_span, psi, method='RK45',
                                            t_eval=None, dense_output=False,
                                            events=is_norm_positive, vectorized=False)
            values=sol.y
            psi=values[ : , -1]
            if len(sol.t_events[0])<1:                              #We reached the final time without jump.
                finished=True
            else:                                                   #There is a jump
                (type,tab)=compute_jump_probas(psi,basis_vector)
                m=get_jump_index(tab)
                (psi,mat2)=quantum_jump(type,basis_vector,psi,mat1,mat2,indices,m-1)

                #Update of the Hamiltonian, time-span and random number
                H_on_psi=get_derivative(mat1,mat2,**kwargs)
                t_span=(sol.t[-1],tf)
                rn=random.random()

    return (psi,mat2)




####JUMP routines###
def get_test_jump(rn):
    """"Decorated. This function returns the function to evaluate for stopping of t-evol."""
    def norm_positive_loc(t,y):
        return np.linalg.norm(y)-rn
    return norm_positive_loc

def get_jump_index(tab):
    """This function returns the index of the jump."""
    rn2=random.random()
    temp=0.
    m=0
    while temp<rn2 and m<len(tab):
        temp+=tab[m]
        m+=1
    return m


def quantum_jump(type,basis_vector,psi_loc,mat1loc,mat2loc,indices,location_jump):
    """This function computes the effect of a quantum jump, returns the new wf and the Ham."""
    if type=="Emission":                                #Jump by spontaneous emission
        (psi_new,indices_to_delete)=jump_elements(basis_vector,psi_loc,indices,location_jump)
        rn=random.random()
        if rn<settings.branching_ratio:                 #If one goes to the uncoupled
            ido=np.identity(mat2loc.shape[0])           #ground state, we set the corresponding
            for a in indices_to_delete:                 #matrix elements to zero, so that further
                ido[a,a]=0.                             #re-excitation will not be possible
            ido=csr_matrix(ido)
            for mm in indices_to_delete:
                mat2loc=ido@mat2loc@ido
        return (psi_new/np.linalg.norm(psi_new),mat2loc)

    else:                                               #Jump by dephasing, here no modification of mat2
        psi_new=dephasing(basis_vector,psi_loc,location_jump)
        return (psi_new/np.linalg.norm(psi_new),mat2loc)

def compute_jump_probas(psi,basis_vector):
    """This function computes the probabilities of jumps, and the type \in {Emission,Dephasing}."""
    tab=np.zeros(settings.N)
    G=settings.Gamma                                                      #taux emission spontanee
    g=settings.gamma_deph                                                 #taux dephasing

    rn3=random.random()                                                   #Determination of the type of event
    if rn3<=G/(g+G):
        type="Emission"
    else:
        type="Dephasing"

    p_tot=0.
    for mm in range(settings.N):                                          #we loop over all the possible jump sites
        H_loc=-1j/2.*sigma_z_operator(basis_vector,mm)                  #Creation of jump operators
        Hpsi=np.multiply(H_loc,psi)
        tab[mm]=abs(np.vdot(psi,1j*Hpsi))                                 #Probability of jump mm
        p_tot+=abs(np.vdot(psi,1j*Hpsi))
    return (type,tab/p_tot)




def jump_elements(basis_vector,psi_loc,indices,location_jump):
    """
    This function will return the new wavefunction after a jump at position location_jump on psi_loc
    It will also return the indices of the Hamiltonian to set to zero after the jump if there is a
    jump towards the uncoupled ground state.
    """
    index_to_delete=[]
    psi_new=np.zeros_like(psi_loc)
    for ii,basis_vector_loc in enumerate(basis_vector):
        if (set([location_jump]).issubset(set(basis_vector_loc))):      #The jump site is part of the target
            continue                                                    #vector --> Not concerned
        (n_initial,n_final)=get_indices(basis_vector_loc,indices)       #Get the indices to look for the parent state
        if n_initial<0. or n_final<0.:                                  #Parent state do not exist.
            continue
        for mm in range(n_initial,n_final):
            if set(basis_vector_loc)|set([location_jump])==set(basis_vector[mm]):
                psi_new[ii]=psi_loc[mm]                                 #we set the target value to the
                index_to_delete.append(mm)                              #parent value, and add the parent
    return (psi_new,index_to_delete)                                    #index to the list for future possible deletion.



def dephasing(basis_vector,psi_loc,location_jump):
    """This function will return the new wavefunction after a dephasing event at position location_jump on psi_loc."""
    psi_new=np.zeros_like(psi_loc)
    for ii,basis_vector_loc in enumerate(basis_vector):
        if (set([location_jump]).issubset(set(basis_vector_loc))):      #The jump site is part of the target
            continue                                                    #vector --> Not concerned
        psi_new[ii]=psi_loc[ii]                                         #else, projection.
    return psi_new






####RUN routines. Different kinds of evolutions. 


def QAOA_single_run_observable_heuristics(theta,H,psi_l,H_1,H_2,H_diss,N_max,N_min,indices,stability_threshold,seuil_prog):
    # We can make it a little bit more modular as well.
    p=int(len(theta))
    val_tab=[]
    for kk in range(N_max):
        psi=psi_l
        mat_second=H_1
        for pp in range(p):
            if pp%2==0:
                mat_first=H_diss
                (psi,mat_second)=evol_scipy(psi,mat_first,mat_second,theta[pp],basis_vector=H,
                                                    indices=indices)
            else:
                if settings.type_evolution=="mixte":
                    mat_first=H_2+H_diss
                    (psi,mat_second)=evol_scipy(psi,mat_first,mat_second,theta[pp],basis_vector=H,
                                                indices=indices)
                else:
                    (psi,mat_second)=evol_scipy(psi,mat_first,mat_second,theta[pp],basis_vector=H,
                                                indices=indices,tunneling='off')
        ###We compute the observable only at the end of the calculation
        psi=psi/np.linalg.norm(psi)
        val_tab.append(compute_observable(H,psi,H_2))#,var_progressive=False,seuil_var_progressive=seuil_prog))
        ##Test if we have gathered enough statistics for the precision threshold that we ask. We also ask a min number of traj
        if np.std(val_tab)/np.sqrt(kk+1.)<stability_threshold and kk>N_min:
            #print(np.mean(val_tab)," ",np.std(val_tab)," ",kk)
            return np.mean(val_tab)
    #print(np.mean(val_tab)," ",np.std(val_tab)," ",kk)
    return np.mean(val_tab)
