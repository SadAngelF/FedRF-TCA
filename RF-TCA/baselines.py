import pickle
from importlib_metadata import FreezableDefaultDict
import numpy as np
import time
import matplotlib.pyplot as plt
from classifier.SVM import SVM_classifier
from utils.datasets import data, resnet_18
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics

from TCA import TCA, vanilla_TCA, R_TCA
from JDA import JDA
from CORAL import CORAL

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

def RF_TCA_run(source_data, source_label, target_data, target_label, save_file = '/data/code/RM_TL/code/results/RF/'):
    n_rf_list = ['']
    m_list = [100]
    sigma_list = list(np.linspace(5, 15, 11))
    mu_list = list(np.logspace(-3, 3, 7))
    kernel_list = ['rbf', 'laplacian', 'cauchy']
    
    acc = []
    time_list = []
    results = pd.DataFrame(columns = ['Source', 'Target', 'n_rf', 'm', 'sigma', 'mu', 'kernel', 'Acc', 'Time'])
    result_name = time.strftime("%Y%m%d-%H-%M", time.localtime())
    num_run = len(n_rf_list) * len(m_list) * len(sigma_list) * len(mu_list) * len(kernel_list)
    runs = 0
    for n_rf in n_rf_list:
        for m in m_list:
            for sigma in sigma_list:
                for mu in mu_list:
                    for kernel in kernel_list:
                        # TCA:
                        s_time = time.time()
                        # T_new = TCA(source_data, target_data, m, sigma, mu, kernel).real
                        T_new = R_TCA(source_data, target_data, 30, 500, 1, 1.5, 'rbf').real
                        # np.savez('T_new.npz', T_new = T_new)
                        # T_new = np.load('T_new.npz')['T_new']
                        
                        # tca_t = time.time() - s_time
                        # TCA_RF:
                        # T_new = RF_TCA(source_data, target_data, m, n_rf, sigma, mu, 'laplacian').real
                        # T_new = Wait_TCA(source_data, target_data, m, n_rf, sigma, 'rbf').real
                        tca_t = time.time() - s_time
                        print("Time:", tca_t)
                        train_data = T_new[:source_data.shape[0], :]
                        test_data = T_new[source_data.shape[0]:,:]

                        classifier = SVM_classifier()
                        classifier.train(train_data, source_label)
                        acc_ = classifier.test(test_data, target_label)
                        print("SVM: ", acc_)
                        
                        # clf = KNeighborsClassifier(n_neighbors=1)
                        # clf.fit(train_data, source_label)
                        # Y_tar_pseudo = clf.predict(test_data)
                        # acc = sklearn.metrics.accuracy_score(target_label, Y_tar_pseudo)
                        # print("KNN: ", acc)

                        acc.append(acc_)
                        time_list.append(tca_t)
                        results = results.append({'Source': Source, 'Target': Target, 'n_rf': n_rf, 'm': m, 'sigma': sigma, 'mu':mu, 'kernel':kernel, 'Acc': acc_, 'Time':tca_t}, ignore_index=True)
                        # print("sigma = %f has finished." % (sigma))
                        runs += 1
                        print(runs, "/", num_run, ": The run of accuracy is ", acc[-1])
                        # print("The time of it is ", tca_t)
                        results.to_csv(save_file + result_name + '.csv')
                        
def TCA_run(source_data, source_label, target_data, target_label, save_file = '/data/code/RM_TL/code/results/'):
    n_rf_list = ['']
    m_list = [50, 100, 150, 200, 250, 300]
    sigma_list = [5] # list(np.linspace(5, 15, 11))
    mu_list = [3] # list(np.logspace(-3, 3, 7))
    kernel_list = ['rbf']
    
    acc = []
    time_list = []
    results = pd.DataFrame(columns = ['Source', 'Target', 'n_rf', 'm', 'sigma', 'mu', 'kernel', 'Acc', 'Time'])
    result_name = time.strftime("%Y%m%d-%H-%M", time.localtime())
    num_run = len(n_rf_list) * len(m_list) * len(sigma_list) * len(mu_list) * len(kernel_list)
    runs = 0
    for n_rf in n_rf_list:
        for m in m_list:
            for sigma in sigma_list:
                for mu in mu_list:
                    for kernel in kernel_list:
                        # TCA:
                        s_time = time.time()
                        # T_new = TCA(source_data, target_data, m, sigma, mu, kernel).real
                        T_new = TCA(source_data, target_data, m, 5, 31.6, 'rbf').real
                        # T_new = vanilla_TCA(source_data, target_data, 30, 1, 1, 'rbf').real
                        # np.savez('T_new.npz', T_new = T_new)
                        # T_new = np.load('T_new.npz')['T_new']
                        
                        # tca_t = time.time() - s_time
                        # TCA_RF:
                        # T_new = RF_TCA(source_data, target_data, m, n_rf, sigma, mu, 'laplacian').real
                        # T_new = Wait_TCA(source_data, target_data, m, n_rf, sigma, 'rbf').real
                        tca_t = time.time() - s_time
                        train_data = T_new[:source_data.shape[0], :]
                        test_data = T_new[source_data.shape[0]:,:]

                        classifier = SVM_classifier()
                        classifier.train(train_data, source_label)
                        acc_ = classifier.test(test_data, target_label)
                        print("SVM: ", acc_)
                        
                        # clf = KNeighborsClassifier(n_neighbors=1)
                        # clf.fit(train_data, source_label)
                        # Y_tar_pseudo = clf.predict(test_data)
                        # acc = sklearn.metrics.accuracy_score(target_label, Y_tar_pseudo)
                        # print("KNN: ", acc)

                        acc.append(acc_)
                        time_list.append(tca_t)
                        results = results.append({'Source': Source, 'Target': Target, 'n_rf': n_rf, 'm': m, 'sigma': sigma, 'mu':mu, 'kernel':kernel, 'Acc': acc_, 'Time':tca_t}, ignore_index=True)
                        runs += 1
                        print(runs, "/", num_run, ": The run of accuracy is ", acc[-1])
                        results.to_csv(save_file + result_name + '.csv')
                        


def JDA_run(source_data, source_label, target_data, target_label, save_file = '/data/code/RM_TL/code/results/JDA/'):
    dim_list = [100]
    lamb_list = list(np.logspace(-3, 3, 7))
    gamma_list = list(np.linspace(5, 15, 11))
    T_list = [10]
    kernel_list = [1] #['linear', 'rbf', 'primal']
    

    acc = []
    time_list = []
    results = pd.DataFrame(columns = ['Source', 'Target', 'dim', 'lamb', 'gamma', 'T', 'kernel', 'Acc', 'Time'])
    result_name = time.strftime("%Y%m%d-%H-%M", time.localtime())
    num_run = len(dim_list) * len(lamb_list) * len(gamma_list) * len(T_list) * len(kernel_list)
    runs = 0
    for dim in dim_list:
        for lamb in lamb_list:
            for gamma in gamma_list:
                for T in T_list:
                    for kernel in kernel_list:
                        # TCA:
                        s_time = time.time()
                        jda = JDA(kernel_type='rbf', dim=100, lamb=0.1, gamma=5)  # lamb is mu, gamma is sigma
                        acc_, ypre, list_acc = jda.fit_predict(source_data, source_label, target_data, target_label)
                        e_time = time.time()
                        t = e_time - s_time
                        print('Time: ', t)
                        print('Acc: ', acc_)
                        acc.append(acc_)
                        time_list.append(t)
                        results = results.append({'Source': Source, 'Target': Target, 'dim': dim, 'lamb': lamb, 'gamma': gamma, 'T':T, 'kernel':kernel, 'Acc': acc_, 'Time':t}, ignore_index=True)
                        runs += 1
                        print(runs, "/", num_run, ": The run of accuracy is ", acc[-1])
                        results.to_csv(save_file + result_name + '.csv')
                        

def CORAL_run(source_data, source_label, target_data, target_label, save_file = '/data/code/RM_TL/code/results/CORAL/'):

    acc = []
    time_list = []
    results = pd.DataFrame(columns = ['Source', 'Target', 'Acc', 'Time'])
    result_name = time.strftime("%Y%m%d-%H-%M", time.localtime())
    runs = 0
    num_run = 1
    
    # TCA:
    s_time = time.time()
    coral = CORAL()
    acc_, ypre = coral.fit_predict(source_data, source_label, target_data, target_label)
    e_time = time.time()
    t = e_time - s_time
    acc.append(acc_)
    time_list.append(t)
    results = results.append({'Source': Source, 'Target': Target, 'Acc': acc_, 'Time':t}, ignore_index=True)
    
    runs += 1
    print(runs, "/", num_run, ": The run of accuracy is ", acc[-1])
    
    results.to_csv(save_file + result_name + '.csv')



if __name__ == '__main__':

    # Read data:
    # dataset1: 'USPS', 'SVHN', 'MNIST', 'MNIST-M'
    # dataset2: 
    #       office10: 
    #               (1) Decaf6: 'office10_decaf6_amazon', 'office10_decaf6_caltech', 'office10_decaf6_dslr', 'office10_decaf6_webcam'
    #               (2) Surf:   'office10_surf_amazon', 'office10_surf_caltech', 'office10_surf_dslr', 'office10_surf_webcam'
    #            #  (3) Raw:   'office10_raw_amazon', 'office10_raw_caltech', 'office10_raw_dslr', 'office10_raw_webcam'
    #       office31:
    #               (1) Decaf6: 'office31_decaf6_amazon', 'office31_decaf6_dslr', 'office31_decaf6_webcam'
    #               (2) Decaf7: 'office31_decaf7_amazon', 'office31_decaf7_dslr', 'office31_decaf7_webcam'
    #            #  (3) Raw:   'office31_raw_amazon', 'office31_raw_dslr', 'office31_raw_webcam'
    Source = 'office10_decaf6_webcam'
    Target = 'office10_decaf6_dslr'
    for Source in ['office10_decaf6_amazon']: # 'office10_decaf6_webcam'
        for Target in ['office10_decaf6_webcam']: # 'office10_decaf6_amazon'
            if Source == Target:
                continue
            data_flag = 0
            if Source in ['USPS', 'SVHN', 'MNIST', 'MNIST-M'] and Target in ['USPS', 'SVHN', 'MNIST', 'MNIST-M']:
                data_flag = 1
            data1 = data(Source, download=False, root_dir = '/data/code/RM_TL/code/data')
            train_data, train_data_label = data1.train_data()
            # train_data = train_data.reshape(train_data.shape[0], -1)/255.0
            # train_data = train_data/255.0
            data2 = data(Target, download=False, root_dir = '/data/code/RM_TL/code/data')
            train_data_m, train_data_m_label = data2.train_data()
            # train_data_m = train_data_m.reshape(train_data_m.shape[0], -1)/255.0
            # train_data_m = train_data_m/255.0

            if data_flag:
                # Choose some data form the whole dataset:
                number_list = range(10)
                source_data, source_label = choose_number(number_list, train_data, train_data_label)
                target_data, target_label = choose_number(number_list, train_data_m, train_data_m_label)

                # resnet18 features:
                source_data, source_label = resnet_18(source_data, source_label)
                target_data, target_label = resnet_18(target_data, target_label)
            else:
                source_data, source_label = train_data, train_data_label
                target_data, target_label = train_data_m, train_data_m_label
            # laten space data transfer
            # RF_TCA_run(source_data, source_label, target_data, target_label, save_file = '/data/code/RM_TL/code/results/RF/')
            TCA_run(source_data, source_label, target_data, target_label, save_file = '/data/code/RM_TL/code/results/TCA/')
            # JDA_run(source_data, source_label, target_data, target_label, save_file = '/data/code/RM_TL/code/results/JDA/')
            # CORAL_run(source_data, source_label, target_data, target_label, save_file = '/data/code/RM_TL/code/results/CORAL/')
            # DeepCORAL