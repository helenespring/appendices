import numpy as np
import scipy
import kwant

def gen_EVS_permzn(mzn,ksp):
    '''
    We want a matrix where each entry is associated to
    a point in 2d k-space. Each point has a vector of 
    EV and ES in orbital basis. mzn is the value of 
    exchange splitting to be fed as a parameter to the 
    system builder, and ksp is the number of points 
    selected per k-axis.
    '''
    sys=make_Ham(m=mzn) #input Kwant system with translational symmetry
    B = np.array(sys.symmetry.periods).T
    A = B @ np.linalg.inv(B.T @ B)
    syst=kwant.wraparound.wraparound(sys).finalized()
    
    def energy_norm(kx, ky):
        #Calculates the set of eigenvalues and eigenstates 
        #at a point (kx,ky) in k-space
        k = np.array([kx, ky])
        kx, ky = np.linalg.solve(A, k)
        H = syst.hamiltonian_submatrix([kx, ky], sparse=False)
        m=scipy.linalg.eigh(H)
        return m 
    
    def k_mats(prec):
        #Generates k-space coordinates
        long_ks=np.linspace(-np.pi, np.pi, prec,endpoint=False) 
        dupl=[1 for i in range(prec)]
        ky_ks=np.kron(long_ks,dupl)
        kx_ks=np.kron(dupl,long_ks)
        k2=prec*prec
        return ky_ks,kx_ks,k2
    
    ky_ks,kx_ks,k2=k_mats(ksp)
    evs=[energy_norm(kx_ks[u], ky_ks[u]) for u in range(k2)] 
    
    ev=[evs[i][0] for i in range(k2)] #vector of eigenvalues
    es=[evs[i][1] for i in range(k2)] #vector of eigenstates
    
    return ev,es
    
def gen_EVS(mznprec,ksp):
    '''
    Here we generate the supermatrices of the EV and ES 
    (array of matrices) for several magnetization values.
    mznprec determines the length of this array.
    '''
    
    if mznprec%2!=0:
        mazn=np.linspace(-2,2,mznprec)
    else:
        mazn=np.linspace(-2,2,mznprec+1) 
    supermatrix=[gen_EVS_permzn(m,ksp=ksp) for m in mazn] 
    ev_supermatrix=[supermatrix[i][0] for i in range(len(mazn))]
    es_supermatrix=[supermatrix[i][1] for i in range(len(mazn))]
    return ev_supermatrix,es_supermatrix
    
    
    
    
    
