import pickle
import numpy as np
from TCA import TCA, vanilla_TCA, R_TCA
import time
import matplotlib.pyplot as plt
from classifier.NN import MNN_classifier
from classifier.SVM import SVM_classifier
from utils.datasets import data, resnet_18, resnet_34
import pandas as pd
import time


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


if __name__ == '__main__':

    # Read data:
    # dataset1: 'USPS', 'SVHN', 'MNIST', 'MNIST-M'
    # dataset2: 
    #       office10: 958, 1123, 157, 295
    #               (1) Decaf6: 'office10_decaf6_amazon', 'office10_decaf6_caltech', 'office10_decaf6_dslr', 'office10_decaf6_webcam'
    #               (2) Surf:   'office10_surf_amazon', 'office10_surf_caltech', 'office10_surf_dslr', 'office10_surf_webcam'
    #               (3) Raw:   'office10_raw_amazon', 'office10_raw_caltech', 'office10_raw_dslr', 'office10_raw_webcam'
    #       office31:
    #               (1) Decaf6: 'office31_decaf6_amazon', 'office31_decaf6_dslr', 'office31_decaf6_webcam'
    #               (2) Decaf7: 'office31_decaf7_amazon', 'office31_decaf7_dslr', 'office31_decaf7_webcam'
    #               (3) Raw:   'office31_raw_amazon', 'office31_raw_dslr', 'office31_raw_webcam'

    index = 0
    for Source in ['office10_decaf6_amazon']: # 'office10_decaf6_webcam'
        for Target in ['office10_decaf6_caltech']: # 'office10_decaf6_amazon'
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
                # source_data, source_label = resnet_34(source_data, source_label)
                # target_data, target_label = resnet_34(target_data, target_label)
            else:
                source_data, source_label = train_data[:500,:], train_data_label[:500,:]
                target_data, target_label = train_data_m[:500,:], train_data_m_label[:500,:]


            # laten space data transfer

            n_rf_list = ['']
            m_list = [50]
            sigma_list = [3] # list(np.linspace(5, 15, 11)) # [13,10,15,5,7,13,6,9,8,13,10,6]
            mu_list = list(np.logspace(-5, 5, 100)) # [1000,1,1000,100,1,1000,100,1000,100,0.01,0.1,10]
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
                                T_new = vanilla_TCA(source_data, target_data, m, sigma, mu, kernel).real
                                # T_new = R_TCA(source_data, target_data, m, 500, sigma, mu, kernel).real
                                tca_t = time.time() - s_time
                                train_data = T_new[:source_data.shape[0], :]
                                test_data = T_new[source_data.shape[0]:,:]


                                classfier = MNN_classifier(m)
                                acc_ = 0
                                for i in range(50):
                                    classfier.train(train_data, source_label, i)
                                    acc__ = classfier.test(test_data, target_label, i)
                                    if acc__>acc_:
                                        acc_ = acc__
                                acc.append(acc_)

                                
                                time_list.append(tca_t)
                                results = results.append({'Source': Source, 'Target': Target, 'n_rf': n_rf, 'm': m, 'sigma': sigma, 'mu':mu, 'kernel':kernel, 'Acc': acc_, 'Time':tca_t}, ignore_index=True)
                                runs += 1
                                print(runs, "/", num_run, ": The run of accuracy is ", acc[-1])
                                results.to_csv('/data/code/RM_TL/code/results/' + result_name + '_MLP.csv')
