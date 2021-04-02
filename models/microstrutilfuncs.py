import numpy as np
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, grad, random, jacfwd, value_and_grad


#%% Mesh

#--------------------------#
def getMeshStructure(mesh):
    # returns edofMat: array of size (numElemsX8) with 
    # the global dof of each elem
    # idx: A tuple informing the position for assembly of computed entries 
    nx, ny = mesh['nelx'], mesh['nely']
    edofMat=np.zeros((nx*ny,8),dtype=int)
    for elx in range(nx):
        for ely in range(ny):
            el = ely+elx*ny
            n1=(ny+1)*elx+ely
            n2=(ny+1)*(elx+1)+ely
            edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2,\
                            2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1]);
    iK = tuple(np.kron(edofMat,np.ones((8,1))).flatten().astype(int))
    jK = tuple(np.kron(edofMat,np.ones((1,8))).flatten().astype(int))
    idx = jax.ops.index[iK,jK]
    return edofMat, idx;
#--------------------------#
def assignMeshDofs(mesh):
    nx, ny = mesh['nelx'], mesh['nely']
    nodenrs = np.arange(0,(1+nx)*(1+ny)).reshape((1+nx,1+ny)).T;
    ncrnrs = np.array([nodenrs[-1,0] ,nodenrs[-1,-1],\
                   nodenrs[0,-1], nodenrs[0,0] ]);
    dcrnrs1 = np.array([[2*n, 2*n+1] for n in ncrnrs]).reshape(1,-1);
    
    # Maybe use logic of nur?
    nlb = np.setdiff1d(np.union1d(np.arange(0, ny+1),\
                      np.arange(ny, (ny+1)*(nx+1), ny+1 )), ncrnrs) ;
    dlb3 = np.array([[2*n, 2*n+1] for n in nlb]).reshape(1,-1);
    

    nur = np.hstack((nodenrs[1:-1,-1].T,nodenrs[0,1:-1]));
    dur4 = np.array([[2*n, 2*n+1] for n in nur]).reshape(1,-1);
    
    nbrd = np.hstack((nur, nlb, ncrnrs))
    nintr = np.setdiff1d(np.arange(0, (1+nx)*(1+ny)), nbrd)
    dintr2 = np.array([[2*n, 2*n+1] for n in nintr]).reshape(1,-1);
    
    dofs = {'interior':dintr2.flatten(), 'corner':dcrnrs1.flatten(),\
         'leftBtm':dlb3.flatten(), 'rightUp':dur4.flatten()};
    return dofs



#%% Material
#--------------------------#
def getD0(matProp):
    # the base constitutive matrix assumes unit 
    #area element with E = 1. and nu prescribed.
    # the material is also assumed to be isotropic.
    # returns a matrix of size (8X8)
    E = 1.
    nu = matProp['nu'];
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,\
                   -1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = \
    E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
    return KE;
#--------------------------#
def getInitialDensity(mesh, vf):
    nx, ny = 1.*mesh['nelx'], 1.*mesh['nely']
    rho = vf*np.ones((mesh['nely'],mesh['nelx']));
    d = np.minimum(nx,ny)/3.;
    ctr = 0;
    for i in range(mesh['nelx']):
        for j in range(mesh['nely']):

            r = np.sqrt(  (i-nx/2. -0.5)**2 + (j-ny/2.-0.5)**2  );
            if(r < d):
                rho[j,i] = vf/2.;
            ctr += 1;
    rho = rho.reshape(-1);
    return rho

#--------------------------#
#%% Filter
def computeFilter(mesh, rmin):
    nx, ny = mesh['nelx'], mesh['nely']
    H = np.zeros((nx*ny,nx*ny));

    for i1 in range(nx):
        for j1 in range(ny):
            e1 = (i1)*ny+j1;
            imin = max(i1-(np.ceil(rmin)-1),0.);
            imax = min(i1+(np.ceil(rmin)),nx);
            for i2 in range(int(imin), int(imax)):
                jmin = max(j1-(np.ceil(rmin)-1),0.);
                jmax = min(j1+(np.ceil(rmin)),ny);
                for j2 in range(int(jmin), int(jmax)):
                    e2 = i2*ny+j2;
                    H[e1, e2] = max(0.,rmin-\
                                       np.sqrt((i1-i2)**2+(j1-j2)**2));

    Hs = np.sum(H,1);
    return H, Hs;
#--------------------------#
def applySensitivityFilter(ft, x, dc, dv):
    if (ft['type'] == 1):
        dc = np.matmul(ft['H'],\
                         np.multiply(x, dc)/ft['Hs']/np.maximum(1e-3,x));
    elif (ft['type'] == 2):
        dc = np.matmul(ft['H'], (dc/ft['Hs']));
        dv = np.matmul(ft['H'], (dv/ft['Hs']));
    return dc, dv;


#%% Boundary condition
#--------------------------#
def getBC(mesh):
    nelx, nely = mesh['nelx'], mesh['nely']
    e0 = np.eye(3);
    ufixed = np.zeros((8,3)); 
    for j in range(3):
        ufixed[2,j], ufixed[3,j]=  np.matmul([[e0[0,j],e0[2,j]/2.],\
                                              [e0[2,j]/2,e0[1,j]]]\
                                  ,np.array([nelx,0]).reshape((2,1)));
        
        ufixed[6,j], ufixed[7,j]=  np.matmul([[e0[0,j],e0[2,j]/2.],\
                                              [e0[2,j]/2,e0[1,j]]]\
                                  ,np.array([0, nely]).reshape((2,1)));
        ufixed[4,j] = ufixed[2,j] + ufixed[6,j];
        ufixed[5,j] = ufixed[3,j] + ufixed[7,j];
    
    
    w1 = np.tile(np.vstack((ufixed[2,:], ufixed[3,:])), (nely-1,1));
    w2 = np.tile(np.vstack((ufixed[6,:], ufixed[7,:])), (nelx-1,1));
    wfixed = np.vstack((w1,w2));
    
    return ufixed, wfixed;

#--------------------------#
def oc(rho, dc, dv, ft, vf):
    l1 = 0; 
    l2 = 1e9;
    x = rho.copy();
    move = 0.2;
    while (l2-l1 > 1e-4):
        lmid = 0.5*(l2+l1);
        dr = np.abs(-dc/dv/lmid);

        xnew = np.maximum(0,np.maximum(x-move,\
                        np.minimum(1,np.minimum(\
                        x+move,x*np.sqrt(dr)))));
        if ft['type'] == 1:
            rho = xnew;
        elif ft['type'] == 2:
            rho = np.matmul(ft['H'],xnew)/ft['Hs'];
        if np.mean(rho) > vf:
            l1 = lmid; 
        else:
            l2 = lmid; 

    change =np.max(np.abs(xnew-x));
    return rho, change;
#--------------------------#      

