from re import L
import numpy as np
from numpy.lib.arraysetops import isin
from scipy.optimize import nnls
from RandomF import RFF_perso
import scipy
from sklearn.metrics.pairwise import laplacian_kernel, chi2_kernel
from sklearn.metrics.pairwise import rbf_kernel
import sklearn.metrics

def TCA_kernel(x, y):
    m = 100
    sigma = 7 
    mu = 1000
    
    # here, X is (d, N), (2, 400)

    n1 = x.shape[0]
    n2 = y.shape[0]
    N = n1 + n2

    X = np.concatenate((x.T, y.T), axis=1)

    L = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (i < n1) and (j < n1):
                L[i, j] = 1/(n1**2)
            elif (i >= n1) and (j >= n1):
                L[i, j] = 1/(n2**2)
            else:
                L[i, j] = -1/(n1*n2)

    H = np.eye(N) - np.ones((N, N)) * (1/N)
    
    
    K = Gauss_Kernel_matrix(X, sigma)

    Matrix = np.matmul(K, L)
    Matrix = np.matmul(Matrix, K)
    Matrix = Matrix + mu * np.eye(N)
    # a = Matrix
    # b = np.matmul(K, H)
    # b = np.matmul(b, K)

    Matrix = np.linalg.inv(Matrix)

    Matrix = np.matmul(Matrix, K)
    Matrix = np.matmul(Matrix, H)
    Matrix = np.matmul(Matrix, K)

    eigvals, eigvecs = np.linalg.eig(Matrix)
    #eigvals, eigvecs = scipy.linalg.eig(b, a)
    ind = np.argsort(-eigvals)
    eigvecs = eigvecs[:, ind[:]]
    eigvals = eigvals[ind[:]]
    W = eigvecs[:, :m]
    x_new = np.matmul(K, W)[:n1, :]
    
    return np.dot(x_new, x_new.T)

def Gauss_Kernel_matrix(X, sigma):
    d = np.diag(np.dot(X.T, X))
    d = d[:, np.newaxis]
    one = np.ones(X.shape[1])
    one = one[:, np.newaxis]
    K = - np.dot(d, one.T) - np.dot(one, d.T) + 2*np.dot(X.T, X)
    #K = np.e**(K / (2*sigma**2))
    K = np.exp(K * sigma )
    return K

def laplacian(x, y, sigma):
    SS = laplacian_kernel(x, x, sigma)
    ST = laplacian_kernel(x, y, sigma)
    TS = laplacian_kernel(y, x, sigma)
    TT = laplacian_kernel(y, y, sigma)
    m1 = np.concatenate((SS, TS), axis=0)
    m2 = np.concatenate((ST, TT), axis=0)
    M = np.concatenate((m1, m2), axis=1)
    return M

def cauchy(x, y, sigma):
    SS = chi2_kernel(x, x, sigma)
    ST = chi2_kernel(x, y, sigma)
    TS = chi2_kernel(y, x, sigma)
    TT = chi2_kernel(y, y, sigma)
    m1 = np.concatenate((SS, TS), axis=0)
    m2 = np.concatenate((ST, TT), axis=0)
    M = np.concatenate((m1, m2), axis=1)
    return M

def combine_data(data1, data2):
    if (isinstance(data1, list)):
        Data = data1[0]
        Data = np.r_[Data, data1[1]]
        for data in data2:
            Data = np.r_[Data, data]
        return Data.T
    else:
        Data = np.concatenate((data1, data2), axis=0)
        return Data.T

def compute_number_samples(data):
    if isinstance(data, list):
        number = 0
        for d in data:
            number += d.shape[0]
        return number
    else:
        return data.shape[0]

def TCA(Source, Target, m, sigma, mu = 1e-8, kernel_name = 'rbf'):
    '''
    Input: 
        Source: A list with two matrix of source_data. For example, the first matrix [d, n11] is the label_1 for source task, and the second matrix [d, n12] is the label_2 for source task.
            'd' is the dimension of one sample
        Target: A list with two matrix of source_data. For example, the first matrix [d, n21] is the label_1 for target task, and the second matrix [d, n22] is the label_2 for target task.
            'd' is the dimension of one sample
        m: The dimension of transformation matrix W, [N, m];
    Ouput:
        W: The transformation matrix W, [N, m].
    '''
    
    X = combine_data(Source, Target)
    # here, X is (d, N), (2, 400)

    #norm
    X /= np.linalg.norm(X, axis=0)
    ##

    n1 = compute_number_samples(Source)
    n2 = compute_number_samples(Target)
    N = n1 + n2

    e = np.vstack((1 / n1 * np.ones((n1, 1)), -1 / n2 * np.ones((n2, 1))))
    L = np.matmul(e, e.T)
    L /= np.linalg.norm(L) #norm
    # L = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         if (i < n1) and (j < n1):
    #             L[i, j] = 1/(n1**2)
    #         elif (i >= n1) and (j >= n1):
    #             L[i, j] = 1/(n2**2)
    #         else:
    #             L[i, j] = -1/(n1*n2)
    #L /= np.linalg.norm(L) #norm
    
    H = np.eye(N) - np.ones((N, N)) * (1/N)
    
    if kernel_name == 'rbf':
        K = Gauss_Kernel_matrix(X, sigma)
        # K = rbf_kernel(X.T, gamma=sigma)
    elif kernel_name == 'laplacian':
        K = laplacian(Source, Target, sigma)
    elif kernel_name == 'cauchy':
        K = laplacian(Source, Target, sigma)
    else:
        print('Please choose from rbf, laplacian or cauchy')
        exit()
    #return K

    # Matrix = np.matmul(K, L)
    # Matrix = np.matmul(Matrix, K)
    # Matrix = Matrix + mu * np.eye(N)
    # # a = Matrix
    # # b = np.matmul(K, H)
    # # b = np.matmul(b, K)

    # Matrix = np.linalg.inv(Matrix)

    # Matrix = np.matmul(Matrix, K)
    # Matrix = np.matmul(Matrix, H)
    # Matrix = np.matmul(Matrix, K)


    a, b = np.linalg.multi_dot([K, L, K.T]) + mu * np.eye(N), np.linalg.multi_dot([K, H, K.T])
    w, V = scipy.linalg.eig(b, a)
    ind = np.argsort(-w)
    # np.save('TCA.npy', V[:, ind[:]])
    # print("Save successfully!")
    W = V[:, ind[:m]]

    # eigvals, eigvecs = np.linalg.eig(Matrix)

    # ind = np.argsort(-eigvals)
    # eigvecs = eigvecs[:, ind[:]]
    # eigvals = eigvals[ind[:]]
    # W = eigvecs[:, :m]
    
    x_new = np.matmul(K, W)
    x_new = x_new.T
    x_new /= np.linalg.norm(x_new, axis=0)
    x_new = x_new.T
    
    #return eigvals, eigvecs

    return x_new

def vanilla_TCA(Source, Target, m, sigma, mu = 1e-8, kernel_name = 'rbf'):
    '''
    Input: 
        Source: A list with two matrix of source_data. For example, the first matrix [d, n11] is the label_1 for source task, and the second matrix [d, n12] is the label_2 for source task.
            'd' is the dimension of one sample
        Target: A list with two matrix of source_data. For example, the first matrix [d, n21] is the label_1 for target task, and the second matrix [d, n22] is the label_2 for target task.
            'd' is the dimension of one sample
        m: The dimension of transformation matrix W, [N, m];
    Ouput:
        W: The transformation matrix W, [N, m].
    '''
    
    X = combine_data(Source, Target)
    # here, X is (d, N), (2, 400)

    #norm
    X /= np.linalg.norm(X, axis=0)
    ##

    n1 = compute_number_samples(Source)
    n2 = compute_number_samples(Target)
    N = n1 + n2

    e = np.vstack((1 / n1 * np.ones((n1, 1)), -1 / n2 * np.ones((n2, 1))))
    L = np.matmul(e, e.T)
    L /= np.linalg.norm(L) #norm
    
    
    H = np.eye(N) - np.ones((N, N)) * (1/N)
    
    if kernel_name == 'rbf':
        K = Gauss_Kernel_matrix(X, sigma)
    elif kernel_name == 'laplacian':
        K = laplacian(Source, Target, sigma)
    elif kernel_name == 'cauchy':
        K = laplacian(Source, Target, sigma)
    else:
        print('Please choose from rbf, laplacian or cauchy')
        exit()
    #return K

    Matrix = np.matmul(K, L)
    Matrix = np.matmul(Matrix, K)
    rank_1 = Matrix / (mu + np.linalg.multi_dot([e.T, K, K, e]))
    
    Matrix = 1/mu*(np.eye(N) - rank_1)
    # Matrix = 1/mu*(np.eye(N)) # No rank_1
    # Matrix = 1/mu*(rank_1) # only rank_1

    Matrix = np.matmul(Matrix, K)
    Matrix = np.matmul(Matrix, H)
    Matrix = np.matmul(Matrix, K)

    eigvals, eigvecs = np.linalg.eig(Matrix)
    # eigvals, eigvecs = scipy.linalg.eig(b, a)
    ind = np.argsort(-eigvals)
    eigvecs = eigvecs[:, ind[:]]
    eigvals = eigvals[ind[:]]
    W = eigvecs[:, :m]
    x_new = np.matmul(K, W)
    x_new = x_new.T
    x_new /= np.linalg.norm(x_new, axis=0)
    x_new = x_new.T
    
    #return eigvals, eigvecs

    return x_new

# def RF_TCA(Source, Target, m, n_features, sigma, mu = 1e-8, kernel = 'rbf'):
#     '''
#     Input: 
#         Source: A list with two matrix of source_data. For example, the first matrix [d, n11] is the label_1 for source task, and the second matrix [d, n12] is the label_2 for source task.
#             'd' is the dimension of one sample
#         Target: A list with two matrix of source_data. For example, the first matrix [d, n21] is the label_1 for target task, and the second matrix [d, n22] is the label_2 for target task.
#             'd' is the dimension of one sample
#         m: The dimension of transformation matrix W, [N, m];
#     Ouput:
#         W: The transformation matrix W, [N, m].
#     '''
#     X = combine_data(Source, Target)
#     X /= np.linalg.norm(X, axis=0)
#     # here, X is (d, N), (2, 400)

#     n1 = compute_number_samples(Source)
#     n2 = compute_number_samples(Target)
#     N = n1 + n2

#     e = np.vstack((1 / n1 * np.ones((n1, 1)), -1 / n2 * np.ones((n2, 1))))
#     L = np.matmul(e, e.T)
#     L /= np.linalg.norm(L) #norm

#     H = np.eye(N) - np.ones((N, N)) * (1/N)

#     RF_K = RFF_perso(sigma, n_features, kernel=kernel)
#     RF_K.fit(X.T)
#     pro = RF_K.transform(X.T).T

#     '''
#     Matrix = np.matmul(pro, L)
#     Matrix = np.matmul(Matrix, pro.T)
#     rank_1 = Matrix / (mu + np.linalg.multi_dot([e.T, pro.T, pro, e]))
    
#     Matrix = 1/mu*(np.eye(2*n_features) - rank_1)
#     # Matrix = 1/mu*(np.eye(2*n_features)) # No rank_1
#     # Matrix = 1/mu*(rank_1) # only rank_1

#     Matrix = np.matmul(Matrix, pro)
#     Matrix = np.matmul(Matrix, H)
#     Matrix = np.matmul(Matrix, pro.T)
#     '''

#     Matrix = np.linalg.multi_dot([pro, L, pro.T, pro, pro.T])
#     mu = 1
#     Matrix = Matrix + mu * np.eye(2*n_features)
#     a, b = Matrix, np.linalg.multi_dot([pro, H, pro.T, pro])
#     Matrix = np.linalg.lstsq(a, b)[0]
#     Matrix = np.linalg.multi_dot([pro.T, Matrix])

#     eigvals, eigvecs = np.linalg.eig(Matrix)
#     #eigvals, eigvecs = scipy.linalg.eig(b, a)
#     ind = np.argsort(-eigvals)
#     eigvecs = eigvecs[:, ind[:]]
#     eigvals = eigvals[ind[:]]
    
#     W = eigvecs[:, :m]
#     K = np.linalg.multi_dot([pro.T, pro])
#     x_new = np.matmul(K, W)
#     x_new = x_new.T
#     x_new /= np.linalg.norm(x_new, axis=0)
#     x_new = x_new.T
#     #return eigvals, eigvecs

#     return x_new

def R_TCA(Source, Target, m, n_features, sigma, mu = 1e-8, kernel = 'rbf'):
    '''
    Input: 
        Source: A list with two matrix of source_data. For example, the first matrix [d, n11] is the label_1 for source task, and the second matrix [d, n12] is the label_2 for source task.
            'd' is the dimension of one sample
        Target: A list with two matrix of source_data. For example, the first matrix [d, n21] is the label_1 for target task, and the second matrix [d, n22] is the label_2 for target task.
            'd' is the dimension of one sample
        m: The dimension of transformation matrix W, [N, m];
    Ouput:
        W: The transformation matrix W, [N, m].
    '''
    X = combine_data(Source, Target)
    X /= np.linalg.norm(X, axis=0)
    # here, X is (d, N), (2, 400)

    n1 = compute_number_samples(Source)
    n2 = compute_number_samples(Target)
    N = n1 + n2

    e = np.vstack((1 / n1 * np.ones((n1, 1)), -1 / n2 * np.ones((n2, 1))))
    L = np.matmul(e, e.T)
    L /= np.linalg.norm(L) #norm

    H = np.eye(N) - np.ones((N, N)) * (1/N)

    RF_K = RFF_perso(sigma, n_features, kernel=kernel)
    RF_K.fit(X.T)
    pro = RF_K.transform(X.T).T

    Matrix = np.matmul(pro, L)
    Matrix = np.matmul(Matrix, pro.T)
    rank_1 = Matrix / (mu + np.linalg.multi_dot([e.T, pro.T, pro, e]))
    
    Matrix = 1/mu*(np.eye(2*n_features) - rank_1)
    # Matrix = 1/mu*(np.eye(2*n_features)) # No rank_1
    # Matrix = 1/mu*(rank_1) # only rank_1

    Matrix = np.matmul(Matrix, pro)
    Matrix = np.matmul(Matrix, H)
    Matrix = np.matmul(Matrix, pro.T)

    eigvals, eigvecs = np.linalg.eig(Matrix)
    #eigvals, eigvecs = scipy.linalg.eig(b, a)
    ind = np.argsort(-eigvals)
    eigvecs = eigvecs[:, ind[:]]
    eigvals = eigvals[ind[:]]
    
    W = eigvecs[:, :m]
    x_new = np.matmul(pro.T, W)
    x_new = x_new.T
    x_new /= np.linalg.norm(x_new, axis=0)
    x_new = x_new.T
    #return eigvals, eigvecs

    return x_new

def RR_TCA(Source, Target, m, sigma, mu = 1e-8, kernel_name = 'rbf'):
    '''
    Input: 
        Source: A list with two matrix of source_data. For example, the first matrix [d, n11] is the label_1 for source task, and the second matrix [d, n12] is the label_2 for source task.
            'd' is the dimension of one sample
        Target: A list with two matrix of source_data. For example, the first matrix [d, n21] is the label_1 for target task, and the second matrix [d, n22] is the label_2 for target task.
            'd' is the dimension of one sample
        m: The dimension of transformation matrix W, [N, m];
    Ouput:
        W: The transformation matrix W, [N, m].
    '''
    
    X = combine_data(Source, Target)
    # here, X is (d, N), (2, 400)

    #norm
    X /= np.linalg.norm(X, axis=0)
    ##

    n1 = compute_number_samples(Source)
    n2 = compute_number_samples(Target)
    N = n1 + n2

    e = np.vstack((1 / n1 * np.ones((n1, 1)), -1 / n2 * np.ones((n2, 1))))
    L = np.matmul(e, e.T)
    L /= np.linalg.norm(L) #norm
    
    
    H = np.eye(N) - np.ones((N, N)) * (1/N)
    
    if kernel_name == 'rbf':
        K = Gauss_Kernel_matrix(X, sigma)
    elif kernel_name == 'laplacian':
        K = laplacian(Source, Target, sigma)
    elif kernel_name == 'cauchy':
        K = laplacian(Source, Target, sigma)
    else:
        print('Please choose from rbf, laplacian or cauchy')
        exit()

    Matrix = np.matmul(K, L)
    Matrix = np.matmul(Matrix, K)
    rank_1 = Matrix / (mu + np.linalg.multi_dot([e.T, K, K, e]))
    
    Matrix = 1/mu*(np.eye(N) - rank_1)
    # Matrix = 1/mu*(np.eye(N)) # No rank_1
    # Matrix = 1/mu*(rank_1) # only rank_1

    Matrix = np.matmul(Matrix, K)
    Matrix = np.matmul(Matrix, H)
    Matrix = np.matmul(Matrix, K)

    eigvals, eigvecs = np.linalg.eig(Matrix)
    # eigvals, eigvecs = scipy.linalg.eig(b, a)
    ind = np.argsort(-eigvals)
    eigvecs = eigvecs[:, ind[:]]
    eigvals = eigvals[ind[:]]
    W = eigvecs[:, :m]
    x_new = np.matmul(K, W)
    x_new = x_new.T
    x_new /= np.linalg.norm(x_new, axis=0)
    x_new = x_new.T
    
    #return eigvals, eigvecs

    return x_new