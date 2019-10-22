
def init():

    global problem
    problem="MIS"
    #problem="MaxCut"
    #problem="Custom"

    global type_graph
    type_graph="Unit Disk"
    #type_graph="Fixed Degree"

    global degree
    degree=3

    global dissipation
    dissipation=False

    global distance
    distance=1. #parameter for UD graphs. Distance below which there is an edge between vertices
    # Maybe change to density ---> more relevant parameter

    global N
    N=14 #number of vertices in the graph
    #p=2# Depth of the QAOA algorithm

    global density
    density=2.6

    global deg
    deg=0# parameter for MaxCut graphs. Degree of a regular graph

    global delta_ee
    delta_ee=-2.

    global delta_rr
    delta_rr=-0.

    global Gamma
    Gamma=0.
    #theta=np.ones(2*(p))

    global gamma_deph
    gamma_deph=0.0

    global type_observable
    type_observable=["cVAR",0.6] # Choose in {["energy",0.],["cVAR",seuil]}. cVAR corresponds to the expected shortfall in the "seuil" better cases.
    #type_observable=["energy",0.]

    global type_evolution
    type_evolution="not mixte"
    #type_evolution="mixte"

    global branching_ratio
    branching_ratio=2./3.

    global delt
    delt=2**(-8)

    global Omega1
    Omega1=1.

    global Omega2
    Omega2=1.

    global stability_threshold
    stability_threshold=0.04

    global N_min_sample_diss
    N_min_sample_diss=1

    global N_max_sample_diss
    N_max_sample_diss=1#000
