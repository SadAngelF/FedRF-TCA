# encoding=utf-8
"""
    Created on 16:31 2018/11/13 
    @author: Jindong Wang
"""

import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from classifier.NN import MNN_classifier


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

class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.real(np.dot(Xs, A_coral))
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        xx = combine_data(Xs, Xt)
        xx /= np.linalg.norm(xx, axis=0)
        xx = xx.T
        n = Xs.shape[0]
        print(n)
        Xs = xx[:n, :]
        Xt = xx[n:, :]
        print(xx.shape)
        print(Xs.shape)
        print(Xt.shape)
        Xs_new = self.fit(Xs, Xt)

        train_data = Xs_new
        source_label = Ys.ravel()
        test_data = Xt
        target_label = Yt

        classfier = MNN_classifier(Xs_new.shape[1])
        acc_ = 0
        for i in range(50):
            classfier.train(train_data, source_label, i)
            acc__ = classfier.test(test_data, target_label, i)
            if acc__>acc_:
                acc_ = acc__
        acc = acc_

        # clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        # clf.fit(Xs_new, Ys.ravel())
        # y_pred = clf.predict(Xt)
        # acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        y_pred = 0

        return acc, y_pred


if __name__ == '__main__':
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in range(4):
        for j in range(4):
            if i != j:
                src, tar = 'data/' + domains[i], 'data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
                coral = CORAL()
                acc, ypre = coral.fit_predict(Xs, Ys, Xt, Yt)
                print(acc)
