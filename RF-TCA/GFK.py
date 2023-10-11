# from itertools import Predicate
from operator import index
from random import sample
from turtle import shape
from matplotlib.pyplot import axis
import numpy as np
from numpy.matlib import repmat
from sklearn.decomposition import PCA
# from utils.datasets import data
import scipy.io as sio
# import bob.math
import scipy
from scipy.linalg import fractional_matrix_power
import time
# from MINST_SVM import Source

def GFK(Source, Target, d):
    '''
    Input:
        Source:A list with two matrix of source_data. For example, the first matrix [d, n11] is the label_1 for source task,
             and the second matrix [d, n12] is the label_2 for source task.
            'd' is the dimension of one sample
        Target:A list with two matrix of source_data. For example, the first matrix [d, n21] is the label_1 for target task,
             and the second matrix [d, n22] is the label_2 for target task.
            'd' is the dimension of one sample
        d: the top d eigenvectors
    Output:
        G: The geodesic flow kernel G, []
    '''
    Ps = my_pca(Source)
    Pt = my_pca(Target)
    
    # Geodesic Flow Kernel
    G = GFK_kernel(Ps, Pt[:,:d])

    # getting transformed data
    s_time = time.time()
    rootG = fractional_matrix_power(G,0.5)
    rootG_time = time.time() - s_time
    print('rootG_time=',rootG_time)
    
    Xs_new = Source @ rootG
    Xt_new = Target @ rootG

    return Xs_new, Xt_new



def GFK_kernel(Ps, Pt):
    '''
    Input: 
        Ps: pca.fit_transform(Xs), which is the result of sorted source data
        Pt: top d vectors of pca.fit_transform(Xt), which is the result of sorted target data        
    Ouput:
        G: The geodesic flow kernel G, []
    '''
    Q = Ps
    N = Q.shape[1]
    dim = Pt.shape[1] # 有区别吗? 意义是什么
    # compute the principal angles
    QPt = np.dot(Q.T, Pt)

    V1, V2, V, Gam, Sig = my_gsvd(QPt[0:dim, :],QPt[dim:, :]) # (A,B) 
    V2 = -V2
    
    theta = np.arccos(np.diagonal(Gam))
    # Equation (6)
    eps = 1e-20
    B1 = np.diag(0.5 * (1 + (np.sin(2 * theta) / (2. * np.maximum(theta, eps)))))
    B2 = np.diag(0.5 * ((np.cos(2 * theta) - 1) / (2 * np.maximum(theta, eps))))
    B3 = B2
    B4 = np.diag(0.5 * (1 - (np.sin(2 * theta) / (2. * np.maximum(theta, eps)))))
    
    # Equation (5)
    B1234 = np.vstack((np.hstack((B1,B2,np.zeros((dim,N-2*dim)))),np.hstack((B3,B4,np.zeros((dim,N-2*dim)))),np.zeros((N-2*dim,N)))) 
    G = Q @ np.vstack((np.hstack((V1,np.zeros((dim,N-dim)))),np.hstack((np.zeros((N-dim,dim)),V2)))) @\
        B1234 @ np.vstack((np.hstack((V1,np.zeros((dim,N-dim)))),np.hstack((np.zeros((N-dim,dim)),V2)))).T @ Q.T
    
    return G

def choose_number(number_list, data, label):
    index = []
    num = 0
    a = [0,0,0,0,0,0,0,0,0,0]
    for i in range(data.shape[0]):
        if label[i] in number_list:
            index.append(i)
            num += 1
            a[label[i]] += 1
        if num == 5000:
            break
    return data[index], label[index]

def split(Y, nPerClass,number_list):
    indexi = []
    index = []
    for i in number_list:
        indexi = np.where(Y==i)
        rn = np.random.permutation(indexi.shape[0])

        index = [index,rn[0:min(nPerClass,indexi.shape[0])]]
    return index

def my_pca(dataMat):
    U, s, Wh = scipy.linalg.svd(dataMat)
    return Wh.T

def my_kernel_knn(Xs_new,Xt_new,Ys,Yt): # (G,Xs,Ys,Xt,Yt)
    diag_Xs = np.diag(Xs_new @ Xs_new.T)    
    diag_Xt = np.diag(Xt_new @ Xt_new.T) # (Xt @ G @ Xt.T)
    diag_Xs = np.asmatrix(diag_Xs)
    diag_Xt = np.asmatrix(diag_Xt)

    dist = repmat(diag_Xs.T,1,Yt.shape[0]) +\
            repmat(diag_Xt,Ys.shape[0],1) -\
             2 * Xs_new @ Xt_new.T          # 2 * Xs * G * Xt.T
    index = np.argmin(dist,axis=0)
    prediction = Ys[index]
    accuracy = np.sum(prediction==Yt) / Yt.shape[0]
    return accuracy

# 根据C和S之间角度相同的关系，实现 gsvd 函数的功能
def my_gsvd(A,B):
    '''
    Description: 
        A = U1_transfer @ np.diag(C_Inver) @ V2h
        B = U2 @ np.diag(S) @ V2h
    Input:
        A: QPt[:dim,:], while dim: the top dim eigenvectors
        B: QPt[dim:,:]
    Output:
        U1_transfer, U2, V2h, np.diag(C_Inver), np.diag(S)
    '''
    U1,C,V1h =  scipy.linalg.svd(A, full_matrices=True,lapack_driver='gesvd') # full_matrices=False,lapack_driver='gesvd'
    U2,S,V2h =  scipy.linalg.svd(B, full_matrices=True,lapack_driver='gesvd')
    
    # change the sort of C
    C_Inver = C[::-1]
    diagC_Inver = np.diag(C_Inver)

    U1_transfer = A @ np.linalg.inv(V2h) @ np.linalg.inv(diagC_Inver) 

    return U1_transfer, U2, V2h, np.diag(C_Inver), np.diag(S)


if __name__ == '__main__':

    # Read data:
    file = sio.loadmat('webcam_SURF_L10.mat')
    Xs = file['fts']
    Ys = file['labels']
    
    file2 = sio.loadmat('dslr_SURF_L10.mat')
    Xt = file2['fts']
    Yt = file2['labels']
   
    d = 20
    G = GFK(Xs,Xt,d)
    acc = my_kernel_knn(G,Xs,Ys,Xt,Yt)

    print('the acc is ',acc)


