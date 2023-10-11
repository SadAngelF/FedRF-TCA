# functions to compute the marginal MMD with rbf kernel
from pandas import array
import torch
import numpy as np


def gaussian_mmd(xs, xt, param, sigma):
    # U1 = [[b.T],[W1]]
    # K = np.vstack((np.hstack((Kss,Kst)),np.hstack((Kst,Ktt))))
    W1 = param[0].detach().numpy()
    b = param[1].detach().numpy()
    U1 = np.concatenate((b.T, W1.T), axis=0)
    xs = xs.detach().numpy()
    xt = xt.detach().numpy()
    n1 = xs.shape[0]
    n2 = xt.shape[0]
    X = np.concatenate((xs.T, np.ones()), axis=0)
    # X = np.concatenate((xs.T, xt.T), axis=1)

    n1 = xs.shape[0]
    n2 = xt.shape[0]
    N = n1 + n2
    L = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (i < n1) and (j < n1):
                L[i, j] = 1/(n1**2)
            elif (i >= n1) and (j >= n1):
                L[i, j] = 1/(n2**2)
            else:
                L[i, j] = -1/(n1*n2)
    
    K = Gauss_Kernel_matrix(X, sigma)
    K = torch.from_numpy(K)
    L = torch.from_numpy(L)
    
    tmp = torch.mm(torch.mm(K.cpu(), L.cpu()), K.T.cpu())
    loss = torch.trace(tmp)
    return loss



def Gauss_Kernel_matrix(X, sigma):
    d = np.diag(np.dot(X.T, X))
    d = d[:, np.newaxis]
    one = np.ones(X.shape[1])
    one = one[:, np.newaxis]
    K = - np.dot(d, one.T) - np.dot(one, d.T) + 2*np.dot(X.T, X)
    #K = np.e**(K / (2*sigma**2))
    K = np.exp(K / (2*sigma**2))
    return K

'''
def GaussianMatrix(Xs, Xt, sigma):
    row,col=Xs.shape
    GassMatrix=np.zeros(shape=(row,row))
    Xs=np.asarray(Xs)
    Xt=np.asarray(Xt)
    i=0
    for v_i in Xs:
        j=0
        for v_j in Xt:
            GassMatrix[i,j]=Gaussian(v_i.T,v_j.T,sigma)
            j+=1
        i+=1
    return GassMatrix

def Gaussian(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))
'''