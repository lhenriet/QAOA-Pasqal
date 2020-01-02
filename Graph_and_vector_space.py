
import random,copy,math
import numpy as np
import settings
import quantum_routines
import scipy.integrate
from scipy.optimize import minimize
import scipy
import igraph
from scipy.sparse import csr_matrix





class Graph:
    """
    Graph used for QAOA,either for MIS or for MaxCut

    Attributes : None, random generation of a graph with the parameters specified by seetings.
    Use igraph to create your own instance of a particular graph.
    """

    def __init__(self):

        self.typegraph=settings.type_graph    #Type of the graph
        self.Nvertices=settings.N             #Number of vertices
        edges=[]                              #List of all the connecting edges under the form [[a,b],[c,d],...] if the vertices a and b (c and d) are connected.
        weights=[]                            #Weight of all of the edges in the tab edges
        tab_individual_links=[]               #At position m, one will find the list of vertices connected to vertex m.
        for m in range(self.Nvertices):       #Initialisation
            tab_individual_links.append([])

        if self.typegraph=="Unit Disk":
            #The positions of the vertices of the graph are chosen at random in [0,1]*[0,1]
            #If the distance between two vertices is smaller than distance, then we define an
            #edge connecting them.
            _positions=[]
            for m in range(self.Nvertices):
                _x=random.random()*(float(settings.N)/settings.density)**0.5
                _y=random.random()*(float(settings.N)/settings.density)**0.5
                _positions.append([_x,_y])
                for n in range(m):
                    if ((_x-_positions[n][0])**2+(_y-_positions[n][1])**2)**0.5<settings.distance:
                        edges.append([n,m])                         # Below a given distance, vertices are connected
                        weights.append(1.)                          # for UD-graphs, all weights are one.
                        tab_individual_links[m].append(n)           # Update of the links tab
                        tab_individual_links[n].append(m)

            self.connections=edges
            self.tab_individual_links=tab_individual_links
            self.weights=weights

            #Now we use a igraph object to describe our graph. This module has several interesting built-in functions.
            g=igraph.Graph()
            g.add_vertices(self.Nvertices)
            for m in range(len(tab_individual_links)):
                for n in range(len(tab_individual_links[m])):
                    g.add_edges([(m,tab_individual_links[m][n])])

            self.independent_sets=g.independent_vertex_sets(min=0, max=0)
            self.igraph_representation=g


        elif self.typegraph=="Fixed Degree":
            #We use the Igraph class to generate a certain kind of graph
            g=igraph.Graph()
            sequence=list(np.ones(self.Nvertices))
            for m in range(len(sequence)):
                sequence[m]=settings.degree
            plap=g.Degree_Sequence(sequence)
            self.igraph_representation=plap
            for e in g.es:
                edges.append(list(e.tuple))
                weights.append(random.random())
            self.connections=edges
            self.weights=weights
            self.independent_sets=g.independent_vertex_sets(min=0, max=0)


    def Divide_non_connected_subgraphs(self):
        """This method divides the initial graph in its disconnected subgraphs.

         The objective is to improve the performances of
        QAOA UD-MIS when the connectivity is low.

        Returns a list of all the disconnected subgraphs
        """
        E=[]
        for mm in range(self.Nvertices): #We loop through all the sites of the graph. val is a bool that asserts if
            _val=False                   #the vertex mm is independent of all the other previous clusters.
            _Dest=[]                     #Dest is a list of all the cluster indices the vertex mm is connected to.
            for kk in range(len(E)):
                for nn in range(len(E[kk])):
                    if mm in self.tab_individual_links[E[kk][nn]]:  #if the vertex mm can be found in the links of any
                        _val=True                                   #vertex nn in the cluster kk, then i) we change the value
                        _Dest.append(kk)                            #of _val to True ii) we add the cluster number kk to Dest.
                        E[kk].append(mm)                            #iii) we add vertex mm to cluster kk.
                        break                                       #As soon as we find such link, break.
            if _val==False:
                E.append([mm])                                      #If the vertex mm cannot be found, then we create
            else:                                                   #a new cluster
                if len(_Dest)>1.:
                    for gg in range(len(_Dest)-1,0,-1):             #If mm has been found in (previously) independent clusters
                        for tt in range(len(E[_Dest[gg]])):         #we merge those clusters.
                            if E[_Dest[gg]][tt] not in E[_Dest[0]]:
                                E[_Dest[0]].append(E[_Dest[gg]][tt])
                        del E[_Dest[gg]]
        return E



def indices_n_exc_basis_states(H):
    """
    This function returns the indices such that list_indices_n_exc[n_0]
    is the smallest index jj for which the vector jj has n_0 excitations.
    """
    list_indices_n_exc=[0]
    nn=0
    for mm in range(len(H)):
        if len(H[mm])>nn:
            list_indices_n_exc.append(mm)
            nn+=1
    list_indices_n_exc.append(len(H))
    return  list_indices_n_exc



def generate_Hilbert_space(graph):
    """
    This function generates the Hilbert space, either from a igraph object
    of from a Graph object as defined above.
    """
    if settings.type_graph=="Unit Disk":
        if isinstance(graph, igraph.Graph):
            H=graph.independent_vertex_sets(min=0, max=0)
        else:
            H=graph.independent_sets
        H.insert(0,())
        indices_nexc_H=indices_n_exc_basis_states(H)
        return (H,indices_nexc_H)
    elif settings.type_graph=="Fixed Degree":
        H=[()]
        ll=list(np.arange(settings.N))
        for r in range(1,settings.N+1):
            uu=list(itertools.combinations(ll, r))
            for o in range(len(uu)):
                H.append(uu[o])
        indices_nexc_H=indices_n_exc_basis_states(H)
        return (H,indices_nexc_H)



def generate_Hamiltonians(Hilbert_space,indices_coupling,**kwargs):
    """
    This function returns the Hamiltonians of QAOA.
    The first one is the mixing Hamiltonian.
    The second-one is the phase-separation Hamiltonian.
    The third one is the dissipation
    """
    if settings.problem=="MIS":
        H_Rabi=settings.Omega1*quantum_routines.sigma_x_operator(Hilbert_space,indices_coupling)
        H_detuning=settings.delta_ee*quantum_routines.sigma_z_operator(Hilbert_space)
        H_dissipation=-1j/2.*(settings.Gamma+settings.gamma_deph)*quantum_routines.sigma_z_operator(Hilbert_space)
    elif settings.problem=="MaxCut":
        H_11=settings.Omega1*quantum_routines.sigma_x_operator(Hilbert_space,indices_coupling)
        graph_instance=kwargs.get('graph_instance',False)
        edges=graph_instance.connections
        weights=graph_instance.weights
        H_22=weights[0]*quantum_routines.sigma_z_z_operator(Hilbert_space,edges[0][0],edges[0][1])
        for m in range(1,len(edges)):
            H_22+=weights[m]*quantum_routines.sigma_z_z_operator(Hilbert_space,edges[m][0],edges[m][1])
        H_dissipation=-1j/2.*(settings.Gamma+settings.gamma_deph)*quantum_routines.sigma_z_operator(Hilbert_space)


    else: ##Custom problem. Needs to specify the unitary.
        pass
    return (csr_matrix(H_Rabi),H_detuning,H_dissipation)
