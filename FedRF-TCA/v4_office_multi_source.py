import sys
sys.path.append("..")

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from data.data import numpyDataset, ForeverDataIterator, MyDataset
import os
import pickle
from torchvision import datasets, transforms
from model.linear_classifier import bottleneck
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD, Adam
import torch
torch.cuda.current_device()
# from v2_federated_model import federated_source_and_target_model
from model.v4_office_multi_source_model import federated_office_source_and_target_model
from TorchRandomF import RFF_perso
# from RandomF import RFF_perso
from torchvision import models, transforms as T
from data.visda_dataset import get_dataset
import time
from tqdm import tqdm
from data.dataset_federated import dataset_federated, office_10_dataset_federated
import random
import numpy as np
import math
# os.environ['max_split_size_mb'] = '2048'

device = torch.device("cuda:2")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

# 获取 accuracy
def wyj_check_y_label_for_voting(y_preds, y_sups, class_names):
    y_sups = y_sups.ravel()
    class_num_list = torch.bincount(y_sups)
    if class_num_list.size(dim=0) < len(class_names):
        class_num_list = torch.cat((class_num_list, torch.zeros(len(class_names)-class_num_list.size(dim=0)).to(y_sups.device))).to(y_sups.device)

    # acc_num = [0] * len(class_names)
    acc_num = torch.zeros(len(class_names)).to(y_sups.device)
    all_acc_num = torch.tensor(0).to(y_sups.device)
    nums = torch.tensor(y_sups.size(dim=0)).to(y_sups.device)
    
    # get every source classifier result on target transfer faeture, 
    # y_preds.size()=>(n, 4, 10),4 is the number of source clients, and 10 is the class num; we first get the pred_class of each source clents; then every target feature got a size=(4,1) result, so we need to squeeze tensor to get a size=(4,) result;
    # 
    _, y_preds = torch.topk(y_preds, 1, dim=2)
    # y_preds = y_preds.ravel()
    y_preds = torch.squeeze(y_preds)
    y_preds, _ =torch.mode(y_preds, dim=1) # y_preds is size=(n,) tensor;

    for i in range(nums):
        if y_preds[i] == y_sups[i]:
            all_acc_num += 1
            acc_num[y_preds[i]] += 1

    prd_acc_list = acc_num.to(torch.float32) / torch.maximum(class_num_list, torch.tensor(1.0).to(y_sups.device))
    mean_class_acc = prd_acc_list.mean()
    
    aug_cls_acc_str = ',  '.join(['{}: {:.3%}'.format(class_names[cls_i], acc_num[cls_i]/class_num_list[cls_i])for cls_i in range(len(class_names))])

    return all_acc_num/nums, mean_class_acc, aug_cls_acc_str 


def wyj_check_y_label(y_preds, y_sups, class_names):
    y_sups = y_sups.ravel()
    class_num_list = torch.bincount(y_sups)
    if class_num_list.size(dim=0) < len(class_names):
        class_num_list = torch.cat((class_num_list, torch.zeros(len(class_names)-class_num_list.size(dim=0)).to(y_sups.device))).to(y_sups.device)

    acc_num = torch.zeros(len(class_names)).to(y_sups.device)
    all_acc_num = torch.tensor(0).to(y_sups.device)
    nums = torch.tensor(y_sups.size(dim=0)).to(y_sups.device)

    _, y_preds = torch.topk(y_preds, 1, dim=1)
    y_preds = y_preds.ravel()

    for i in range(nums):
        if y_preds[i] == y_sups[i]:
            all_acc_num += 1
            acc_num[y_preds[i]] += 1

    prd_acc_list = acc_num.to(torch.float32) / torch.maximum(class_num_list, torch.tensor(1.0).to(y_sups.device))
    mean_class_acc = prd_acc_list.mean()
    
    aug_cls_acc_str = ',  '.join(['{}: {:.3%}'.format(class_names[cls_i], acc_num[cls_i]/class_num_list[cls_i])for cls_i in range(len(class_names))])

    return all_acc_num/nums, mean_class_acc, aug_cls_acc_str

def accuracy(output, target):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = 1 # max(topk)
        batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        # _, labels = target.topk(maxk, 1, True, True)
        _, pred = torch.topk(output, 1, dim=1)
        pred = pred.flatten()
        # pred = pred.t()
        # print('pred.size() = ', pred.size())
        # print('target[None].size() = ', target[None].size())
        correct = pred.eq(target)
        
        correct_k = correct.flatten().sum(dtype=torch.float32)
        res = (correct_k * (100.0 / batch_size))

        return res

def digit_five_ForeverDataIterator(source_dataset1, source_dataset2, source_dataset3,  target_dataset, batch_ns, batch_nt):
    source_loader1 = torch.utils.data.DataLoader( source_dataset1, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    source_loader2 = torch.utils.data.DataLoader( source_dataset2, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    source_loader3 = torch.utils.data.DataLoader( source_dataset3, 
                                            batch_size=batch_ns, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    target_loader = torch.utils.data.DataLoader( target_dataset, 
                                            batch_size=batch_nt, 
                                            num_workers=4, 
                                            shuffle=True, 
                                            drop_last=True)
    return [source_loader1, source_loader2, source_loader3, target_loader]               

def calculate_Wrf_matrix(s_Sigma_Sigma_T_sum, s_Sigma_y_sum, t_Sigma_Sigma_T_sum, t_Sigma_y_sum, n_s, n_t, gamma, m, batch_ns):
    # n_s, n_t are num of batch in dataloader
    # all is torch.tensor
    random_feature_dim = s_Sigma_y_sum.size(dim=0)
    Sigma_1 = batch_ns * (s_Sigma_y_sum + t_Sigma_y_sum)
    Sigma_y = (1/n_s) * s_Sigma_y_sum - (1/n_t) * t_Sigma_y_sum
    rank_1 = gamma + torch.mm(Sigma_y.T, Sigma_y)
    Matrix = (torch.eye(random_feature_dim).to(device) - torch.mm(Sigma_y, Sigma_y.T) / rank_1 )

    Sigma_H_Sigma_T = (s_Sigma_Sigma_T_sum + t_Sigma_Sigma_T_sum) - 1/(batch_ns*(n_s+n_t)) * torch.mm(Sigma_1, Sigma_1.T) # since t_Sigma_y_sum is negative
    Matrix = torch.mm(Matrix, Sigma_H_Sigma_T)

    # teh eigenvector matrix of the Matrix
    eigvals, eigvecs = torch.linalg.eig(Matrix)
    eigvals = torch.real(eigvals)
    _, indices = torch.sort(-eigvals)  # 降序排列
    eigvecs = eigvecs[:, indices]
    W_rf = eigvecs[:, :m]
    return W_rf

def get_all_init_W_rf(Sigma_Sigma_T_list, Sigma_y_list, ns_1, ns_2, ns_3, ns_4, nt, gamma, m, batch_ns):
    W_rf_s1 = calculate_Wrf_matrix(Sigma_Sigma_T_list[0], Sigma_y_list[0], Sigma_Sigma_T_list[4], Sigma_y_list[4], n_s=ns_1, n_t=nt, gamma=mu, m=m, batch_ns=batch_ns)    
    W_rf_s2 = calculate_Wrf_matrix(Sigma_Sigma_T_list[1], Sigma_y_list[1], Sigma_Sigma_T_list[4], Sigma_y_list[4], n_s=ns_2, n_t=nt, gamma=mu, m=m, batch_ns=batch_ns)    
    W_rf_s3 = calculate_Wrf_matrix(Sigma_Sigma_T_list[2], Sigma_y_list[2], Sigma_Sigma_T_list[4], Sigma_y_list[4], n_s=ns_3, n_t=nt, gamma=mu, m=m, batch_ns=batch_ns)
    W_rf_s4 = calculate_Wrf_matrix(Sigma_Sigma_T_list[3], Sigma_y_list[3], Sigma_Sigma_T_list[4], Sigma_y_list[4], n_s=ns_4, n_t=nt, gamma=mu, m=m, batch_ns=batch_ns)

    all_Source_Sigma_Sigma_T = Sigma_Sigma_T_list[0] + Sigma_Sigma_T_list[1] + Sigma_Sigma_T_list[2] + Sigma_Sigma_T_list[3]
    all_Source_Sigma_y = Sigma_y_list[0] + Sigma_y_list[1] + Sigma_y_list[2] + Sigma_y_list[3]
    W_rf_t = calculate_Wrf_matrix(all_Source_Sigma_Sigma_T, all_Source_Sigma_y, Sigma_Sigma_T_list[4], Sigma_y_list[4], n_s=(ns_1+ns_2+ns_3+ns_4), n_t=nt, gamma=mu, m=m, batch_ns=batch_ns)
    return W_rf_s1, W_rf_s2, W_rf_s3, W_rf_s4, W_rf_t
    
if __name__ == '__main__':
    # data loading
    # dataset_office: "digit_five", "visda", "office31", "office_caltech_10" 
    #                 "mnist", "svhn", "usps";
    # office: "amazon", "webcam", "dslr", "caltech"
    # "digit_five": "mnist", "usps", "svhn", "mnist-m", "synthetic_digits" 

    dataset_name = "digit_five"
    class_num = 10
    # target_list = ["amazon", "webcam"]
    # target_list = ["dslr", "caltech"]
    target_list = ["amazon", "webcam", "dslr", "caltech"]
    dataset_list = ["amazon", "webcam", "dslr", "caltech"]

    m = 800
    n_features = 1000
    # sigma = 1
    mu = 10
    kernel = 'rbf'

    batch_ns = 128
    batch_nt = batch_ns
    # for i in range(10):
    for target in target_list:
        source_dataset_list = dataset_list.copy()
        source_dataset_list.remove(target)
        # loading datasets
        # source_dataset1, target_dataset, num_classes, class_names = dataset_federated(dataset_name, source1, target)    
        
        # Setup output
        data = time.strftime("%Y%m%d-%H-%M", time.localtime())
        exp = data + 'Average_Wrf_Classifier'
        log_file = 'office_Occasionally_randomset_Average1_n_Wrf_Classifier_sigmalist_droplast_all_to_{}_allpair_{}.txt'.format(target, exp) # _

        if log_file is not None:
            if os.path.exists(log_file):
                print('Output log file {} already exists'.format(log_file))
        def log(text, end='\n'):
            print(text)
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(text + end)
                    f.flush()
                    f.close() 
                    
        if dataset_name == "digit_five":
            num_classes = 10
            class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
        source_dataset1 = office_10_dataset_federated(source_dataset_list[0])
        source_dataset2 = office_10_dataset_federated(source_dataset_list[1])
        source_dataset3 = office_10_dataset_federated(source_dataset_list[2])
        target_dataset = office_10_dataset_federated(target)
        ns_1, ns_2, ns_3, nt = len(source_dataset1), len(source_dataset2), len(source_dataset3), len(target_dataset)
        # ns_1, ns_2, ns_3, ns_4, nt = ns_1-ns_1%batch_ns, ns_2-ns_2%batch_ns, ns_3-ns_3%batch_ns, ns_4-ns_4%batch_ns, nt-nt%batch_nt
        ns_sum = ns_1 + ns_2 + ns_3
        ns_list = [ns_1, ns_2, ns_3, nt]

        # DataLoader and ForeverDataIterator
        # loader_list = [source_loader1, source_loader2, source_loader3, source_loader4, target_loader]
        loader_list = digit_five_ForeverDataIterator(source_dataset1, source_dataset2, source_dataset3, target_dataset, batch_ns, batch_nt)
        train_source_iter1 = ForeverDataIterator(loader_list[0]) # source_loader1
        train_source_iter2 = ForeverDataIterator(loader_list[1]) # source_loader2
        train_source_iter3 = ForeverDataIterator(loader_list[2]) # source_loader3
        train_target_iter = ForeverDataIterator(loader_list[3])# target_loader

        feature_dim = 2048 # 25088
        highest_acc = []
        highest_ave_acc = []
        trade_off1 = 1
        trade_off2 = 1
        # sigma_list = list(np.linspace(0.1, 1, 10))
        # sigma_list = list(np.linspace(1, 11, 6))
        # sigma_list = list(np.linspace(10, 17, 8))
        # sigma_list = {"amazon":5, "webcam":7, "dslr":5, "caltech":5}
        # sigma_list = {"amazon":1, "webcam":1, "dslr":3, "caltech":3}
        sigma_list = {"amazon":1, "webcam":1, "dslr":1, "caltech":1}
        sigma = sigma_list[target]
        # sigma = 1
        log("Multi-Source Setting: Federated Resnet with only (7) layer require_grad=True, learning_rate = 2e-4")
        log("Occasionally Average Classifeir iter_num % 15 (every epoch average twice Classifiers).Just random subset iteration with both 1_n Classifier Averaging and W_rf Averaging")
        # log("Just random subset iteration with both 1_n Classifier Averaging and W_rf Averagin.")
        # log('all clients with both 1_n Classifier Averaging and W_rf Averaging')
        log('dataset = {}, \t source1 = {}, \t source2 = {}, \t source3 = {}, \t  target = {}'.format(dataset_name, source_dataset_list[0], source_dataset_list[1], source_dataset_list[2], target))
        
        # model_weight_path = "/home/wyj/VGG_SVD_exp/weights/digit_all2{}_1_n_Averaging_sigmalist_model_random2.pth".format(target)

        epoches = 50
        iter_num = 20 # 100 # 50 # 200 # 30

        for exp_i in range(10):
            log("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            log('sigma = {}'.format(sigma))
            # model setting
            classifier = federated_office_source_and_target_model(feature_dim=feature_dim, m=m, n_features=n_features, sigma=sigma, gamma=mu, kernel=kernel, class_num=class_num).to(device)
            value_loss = nn.CrossEntropyLoss()
            learning_rate = 2e-4


            # log setting message
            model_setting = 'resnet50'
            log('Model: {} '.format(model_setting))
            log('Setting: m={m}, n_features={n_features}, sigma={sigma}, mu={mu}, kernel={kernel}'.format(m=m, n_features=n_features, sigma=sigma, mu=mu, kernel=kernel))
            log('       :  learning_rate={lr}, batch_ns={batch_ns}, batch_nt={batch_nt}, iter_num={iter_num}, trade_off1={trade_off1}, '.format(lr=learning_rate, batch_ns=batch_ns, batch_nt=batch_nt, iter_num=iter_num, trade_off1=trade_off1))
            optimizer = Adam(classifier.parameters(), lr=learning_rate)
            # classifier.load_state_dict(torch.load(model_weight_path))

            max_acc = torch.tensor(0).to(device)
            max_ace_acc = torch.tensor(0).to(device)
            mmd_average_list = [1, 1 ,1, 1]
            
            # train
            for epoch in range(epoches):
                # if epoch % 25 == 0:
                #     train_source_iter1, train_source_iter2, train_source_iter3, train_source_iter4, train_target_iter = digit_five_ForeverDataIterator(source_dataset1, source_dataset2, source_dataset3, source_dataset4, target_dataset, batch_ns, batch_nt)

                classifier.train()
                source_acc = torch.tensor(0).to(device)
                tar_acc = torch.tensor(0).to(device)
                source_num = torch.tensor(4).to(device)
                mmd_count_list = [0, 0 ,0, 0]
                mmd_sum_list = [0, 0 ,0, 0]
                for iter_i in tqdm(range(iter_num)):
                    x_s1, labels_s1 = next(train_source_iter1)
                    x_s2, labels_s2 = next(train_source_iter2)
                    x_s3, labels_s3 = next(train_source_iter3)
                    x_t, labels_t = next(train_target_iter)
                    x_s1 = x_s1.to(device)
                    x_s2 = x_s2.to(device)
                    x_s3 = x_s3.to(device)
                    x_t = x_t.to(device)

                    labels_s1 = labels_s1.to(torch.int64)
                    labels_s2 = labels_s2.to(torch.int64)
                    labels_s3 = labels_s3.to(torch.int64)
                    labels_t = labels_t.to(torch.int64)
                    labels_s1 = labels_s1.to(device)
                    labels_s2 = labels_s2.to(device)
                    labels_s3 = labels_s3.to(device)
                    labels_t = labels_t.to(device)      

                    source1_Sigma_yi, source2_Sigma_yi, source3_Sigma_yi, target_Sigma_yi, ys1, ys2, ys3, yt = classifier(x_s1, x_s2, x_s3, x_t)
                    # soft voting
                    # yt = (ns_1/ns_sum) * yt[:,0,:] + (ns_2/ns_sum) * yt[:,1,:] + (ns_3/ns_sum) * yt[:,2,:] + (ns_4/ns_sum) * yt[:,3,:]

                    cls_loss1 = value_loss(ys1, torch.flatten(labels_s1, start_dim=0))
                    cls_loss2 = value_loss(ys2, torch.flatten(labels_s2, start_dim=0))
                    cls_loss3 = value_loss(ys3, torch.flatten(labels_s3, start_dim=0))
                    
                    # mmd_loss calculating
                    temp_target_Sigma_yi = target_Sigma_yi.detach_()
                    temp_source1_Sigma_yi = source1_Sigma_yi.detach_()
                    temp_source2_Sigma_yi = source2_Sigma_yi.detach_()
                    temp_source3_Sigma_yi = source3_Sigma_yi.detach_()                    
                    mmd_loss_source1 = classifier.mmd_loss_source1(source1_Sigma_yi, temp_target_Sigma_yi) 
                    mmd_loss_source2 = classifier.mmd_loss_source2(source2_Sigma_yi, temp_target_Sigma_yi)
                    mmd_loss_source3 = classifier.mmd_loss_source3(source3_Sigma_yi, temp_target_Sigma_yi)

                    all_temp_source_Sigma_yi = [temp_source1_Sigma_yi, temp_source2_Sigma_yi, temp_source3_Sigma_yi ]
                    source_mmd_loss_list = [mmd_loss_source1, mmd_loss_source2, mmd_loss_source3]

                    # get a random subset of 4 source domains
                    passing_num1 = torch.tensor(random.randint(0, 2)).to(device)
                    random_i_list = random.sample(range(3), passing_num1)
                    
                    cls_loss = cls_loss1 + cls_loss2 + cls_loss3
                    all_temp_source_Sigma_yi = sum([all_temp_source_Sigma_yi[random_i] for random_i in random_i_list])
                    source_mmd_loss = sum([source_mmd_loss_list[random_i] for random_i in random_i_list])
                    mmd_loss_target = classifier.mmd_loss_target(all_temp_source_Sigma_yi, target_Sigma_yi)
                    
                    target_loss = trade_off2 * mmd_loss_target
                    loss = cls_loss + source_mmd_loss + target_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()                

                    cls_acc1 = accuracy(ys1, labels_s1)
                    cls_acc2 = accuracy(ys2, labels_s2)
                    cls_acc3 = accuracy(ys3, labels_s3)
                    trans_cls_acc, _, _cls_str = wyj_check_y_label(yt, labels_t, class_names)

                    tar_acc = tar_acc + trans_cls_acc
                    if trans_cls_acc > max_acc:
                        max_acc = trans_cls_acc
                    """"""
                    if random_i_list != []:
                        classifier.FedAvg_transfer_layer(random_i_list) # 
                        # , ns_list=exp_mmd_list, nt=sum(exp_mmd_list)
                        # classifier.FedAvg_pair_transfer_layer(passing_num2)

                        # update source classifiers
                        if iter_i % 20 == 0:
                            classifier.FedAvg_classifier(random_i_list) # 
                        # , ns_list=exp_mmd_list

                        # only classifier model 
                        # classifier.FedAvg_only_target_classifier()
                        # losssum, weight_0, weight_1, weight_2, weight_3 = classifier.FedAvg_only_target_classifier(ns_1, ns_2, ns_3, ns_4)

                ave_trans_acc = tar_acc / iter_num
                if ave_trans_acc > max_ace_acc:
                    max_ace_acc = ave_trans_acc

                # evaluate

                # all_acc_point, mean_class_acc, cls_acc_str = wyj_check_y_label_for_voting(yt, labels_t, class_names)
                all_acc_point, mean_class_acc, cls_acc_str = wyj_check_y_label(yt, labels_t, class_names)

                log('epoch = {} , \t cls_acc1({}) = {:.5}, cls_acc2({}) = {:.5}, cls_acc3({}) = {:.5}'.format(epoch, source_dataset_list[0], cls_acc1.item(), source_dataset_list[1], cls_acc2.item(), source_dataset_list[2], cls_acc3.item()))
                log('\t TARGET: mean acc={:.3%}, mean class acc={:.3%}'.format(all_acc_point, mean_class_acc))
                log('\t \t: this_iteration_avg_ace_acc={}'.format(ave_trans_acc))
                log('\t \t: max_avg_ace_acc={}'.format(max_ace_acc))
                log('\t   per class:  {}'.format(cls_acc_str))      
                log(' source_mmd_loss1 = {}, \t source_mmd_loss2 = {}, \t source_mmd_loss3 = {}, \t target_mmd_loss = {} '.format(mmd_loss_source1.item(), mmd_loss_source2.item(), mmd_loss_source3.item(), mmd_loss_target.item())) 
                # log('\t   cls_loss  = {:.6} \t source_mmd_loss1 = {}, \t source_mmd_loss2 = {}, \t source_mmd_loss3 = {}, \t source_mmd_loss4 = {}, \t target_mmd_loss = {} '.format(cls_loss.item(), mmd_average_list[0], mmd_average_list[1], mmd_average_list[2], mmd_average_list[3], mmd_loss_target.item()))
                # log('losssum = {} ,  weight_0 = {} , \t weight_1 = {} , \t weight_2 = {}, \t weight_3 = {} '.format(losssum, weight_0, weight_1, weight_2, weight_3))
                # log('\t   cls_loss  = {:.6} \t source_mmd_loss = {:4.5},  \t target_mmd_loss = {:4.5} '.format(cls_loss.item(), mmd_loss_source.item(), mmd_loss_target.item()))

                # if max_ace_acc > domain_acc_dic[target]:
                #     domain_acc_dic[target] = max_ace_acc
                #     torch.save(classifier.state_dict(), model_weight_path)
                

            highest_acc.append(max_acc.item())
            highest_ave_acc.append(max_ace_acc.item())
            log('highest_acc = {}, \t highest_ave_acc = {}'.format(highest_acc, highest_ave_acc))

        
