import numpy as np

def berry_curvature(evsmat,mzn,b_n,level=0):
    '''
    evsmat is the eigenvalue and eigenstate vector. mzn is 
    the index of exchange splitting. b_n selects a scipy band. 
    level is the energy level below which we want to include 
    berry curvature contributions.
    '''
    
    ev,es=evsmat
    
    ev=ev[mzn]#all k-points, w all bands
    prec=int(np.sqrt(len(ev)))
    k2=len(ev) 
    ev=[ev[i][b_n] for i in range(k2)] 
    ev=np.array(ev)    
    
    def squares(v1,v2,v3,v4):
        #Check if the sum of a square is above 
        #or below the Fermi level
        sumev=ev[v1]+ev[v2]+ev[v3]+ev[v4]
        [new]=np.where(sumev<level)
        #Only select the top-left corner of each 
        #sub-level square
        return v1[new]
    
    newnl=squares(nl,nl_x,nl_y,nl_xy)
    newar=squares(ar,ar_x,ar_y,ar_xy)
    newad=squares(ad,ad_x,ad_y,ad_xy)
    
    t_v=es[mzn]#all k-points, w all bands
    t_v=[t_v[i][:,b_n:b_n+1] for i in range(k2)] 
    t_v=np.array(t_v)
    
    def b_c(ind,indx,indy,indxy):
        #inner products of eigenstates on a square
        fa_1=np.sum(t_v[ind].conj()*t_v[indx],1)
        fa_2=np.sum(t_v[indx].conj()*t_v[indxy],1)
        fa_3=np.sum(t_v[indxy].conj()*t_v[indy],1)
        fa_4=np.sum(t_v[indy].conj()*t_v[ind],1)
        fa=fa_1*fa_2*fa_3*fa_4
        pn=np.angle(fa)
        #Take the principle argument
        pn=np.mod(pn+np.pi/2,np.pi)-np.pi/2
        pn=np.transpose(pn)[0]
        return pn
    
    #inner products away from the periodic boundary cuts
    pn_nl=b_c(nl,nl_x,nl_y,nl_xy)
    #inner products on the (kx=pi-epsilon) boundary
    pn_ar=b_c(ar,ar_x,ar_y,ar_xy)
    #inner products on the (ky=pi-epsilon) boundary
    pn_ad=b_c(ad,ad_x,ad_y,ad_xy)
    #point where the boundaries meet
    last=b_c([k2-1],[ad[0]],[ar[0]],[nl[0]])
    
    #shape into a matrix for use in visual representations
    vect=np.zeros(k2)
    vect[nl]=pn_nl
    vect[ar]=pn_ar
    vect[ad]=pn_ad
    vect[-1]=last
    mat=np.reshape(vect,(prec,prec))
    
    #the sum on the plane gives the conductivity
    return mat,sum(vect)