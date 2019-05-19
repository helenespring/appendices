import numpy as np

def gen_spaces():
    '''
    Generate coordinate map of k-space.
    '''
    ks=int(np.sqrt(len(evsmat[0][0])))
    limit_right=[pp*ks-1 for pp in range(ks)]
    limit_down=[ks*(ks)+i for i in [-ks+j for j in range(ks-1)]]
    ar=np.array([iks for iks in range(ks*ks) if iks 
                 in limit_right])
    ad=np.array([iks for iks in range(ks*ks) if iks 
                 in limit_down])
    nl=np.array([iks for iks in range(ks*ks) if iks 
                 not in limit_right and iks not in limit_down and 
    iks != (ks*ks-1)])
    nl_x=nl+1
    nl_y=nl+ks
    nl_xy=nl+ks+1
    ad_y=ad-ks*(ks-1)
    ad_xy=ad-ks*(ks-1)+1
    ad_x=ad+1
    ar_y=ar+ks
    ar_x=ar-ks+1
    ar_xy=ar+1
    return ar,ad,nl,nl_x,nl_y,nl_xy,ad_x,ad_y,ad_xy,ar_x,ar_y,ar_xy