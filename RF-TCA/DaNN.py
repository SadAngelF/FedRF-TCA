from ast import Num
import DaNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import data_loader
import mmd

import random 
import time
from utils.datasets import data, resnet_18
import mmd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.005 # 0.02
MOMEMTUN = 0.05
L2_WEIGHT = 0.003
DROPOUT = 0.5
N_EPOCH = 2 # 900
BATCH_SIZE = [64, 64]
LAMBDA = 0.25
GAMMA = 10 ^ 3
RESULT_TRAIN = []
RESULT_TEST = []
# surf feature num
FEATURE_NUM = 800 # decaf6=4096;surf=800
log_train = open('log_train_a-w.txt', 'w')
log_test = open('log_test_a-w.txt', 'w')

def mmd_loss(x_src, x_tar):
    return mmd.mix_rbf_mmd2(x_src, x_tar, [GAMMA]) 
    # return DeepDA_mmd.forward(x_src, x_tar)

def last_batch_append(list_tar):
    _, (x_tar, y_target) = list_tar[-1]
    batch_num = _
    rnint = random.randint(0,batch_num-1)
    _append, (x_tar_append, y_target_append) = list_tar[rnint]
    comp_x_tar = torch.cat((x_tar,x_tar_append[0:BATCH_SIZE[0]-x_tar.shape[0]]),0)
    comp_y_tar = torch.cat((y_target,y_target_append[0:BATCH_SIZE[0]-x_tar.shape[0]]),0)

    return comp_x_tar, comp_y_tar

def add_last_batch(data, lable):
    num = data.shape[0]
    batch_num = num // BATCH_SIZE[0]
    last_num = num % BATCH_SIZE[0]
    rnint = random.randint(0,batch_num-1)
    add_num = BATCH_SIZE[0] - last_num
    start_num = rnint*BATCH_SIZE[0]
    data = np.vstack((data,data[start_num:start_num+add_num]))
    lable = np.vstack((lable,lable[start_num:start_num+add_num]))

    return data, lable


def train(model, optimizer, epoch, data_src, data_tar):
    total_loss_train = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    # wyj
    batch_i = 0

    batch_j = 0
    # list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))    
    source_data,source_lable = data_src
    target_data,target_lable = data_tar
    train_num = max(source_data.shape[0],target_data.shape[0]) // BATCH_SIZE[0]

    for i in range(train_num):
        # batch_id, (data, target) = list_src[batch_i]

        data = torch.from_numpy(source_data[batch_i*BATCH_SIZE[0]:(batch_i+1)*BATCH_SIZE[0]]).to(torch.float32)
        target = torch.from_numpy(source_lable[batch_i*BATCH_SIZE[0]:(batch_i+1)*BATCH_SIZE[0]]).to(torch.long)

        # _, (x_tar, y_target) = list_tar[batch_j]
        x_tar = torch.from_numpy(target_data[batch_j*BATCH_SIZE[0]:(batch_j+1)*BATCH_SIZE[0]]).to(torch.float32)
        y_target = torch.from_numpy(target_lable[batch_j*BATCH_SIZE[0]:(batch_j+1)*BATCH_SIZE[0]]).to(torch.long)        

        data, target = data.view(-1,FEATURE_NUM).to(DEVICE), target.view(BATCH_SIZE[0]).to(DEVICE)
        x_tar, y_target = x_tar.view(-1,FEATURE_NUM).to(DEVICE), y_target.view(BATCH_SIZE[0]).to(DEVICE)       
        model.train()
        y_src, x_src_mmd, x_tar_mmd = model(data, x_tar)
        loss_c = criterion(y_src, target)
        loss_mmd = mmd_loss(x_src_mmd, x_tar_mmd)
        pred = y_src.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss = loss_c + LAMBDA * loss_mmd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_train += loss.data
        res_i = 'Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}'.format(
            epoch, N_EPOCH, i+1, train_num, loss.data
        )
        batch_i += 1
        batch_j += 1
        if batch_j >= target_data.shape[0]//BATCH_SIZE[0]:
            batch_j = 0
        if batch_i >= source_data.shape[0]//BATCH_SIZE[0]:
            batch_i = 0
    total_loss_train /=source_data.shape[0]/BATCH_SIZE[0]+1
    acc = correct * 100. / source_data.shape[0] # len(data_src.dataset)
    res_e = 'Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        epoch, N_EPOCH, total_loss_train, correct, source_data.shape[0], acc
    )
    tqdm.write(res_e)
    log_train.write(res_e + '\n')
    RESULT_TRAIN.append([epoch, total_loss_train, acc])
    return model


def test(model, data_tar, e):
    total_loss_test = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    target_data,target_lable = data_tar
    train_num = target_data.shape[0] // BATCH_SIZE[0]
    with torch.no_grad():
        for i in range(train_num):
            # batch_id, (data, target) in enumerate(data_tar):
            data = torch.from_numpy(target_data[i*BATCH_SIZE[0]:(i+1)*BATCH_SIZE[0]]).to(torch.float32)
            target = torch.from_numpy(target_lable[i*BATCH_SIZE[0]:(i+1)*BATCH_SIZE[0]]).to(torch.long)

            data, target = data.view(-1,FEATURE_NUM).to(DEVICE),target.view(BATCH_SIZE[0]).to(DEVICE)
            model.eval()
            ypred, _, _ = model(data, data)
            loss = criterion(ypred, target)
            pred = ypred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            total_loss_test += loss.data
        accuracy = correct * 100. / target_data.shape[0] # len(data_tar[0]) 
        res = 'Test: total loss: {:.6f}, correct: [{}/{}], testing accuracy: {:.4f}%'.format(
            total_loss_test, correct, target_data.shape[0], accuracy # len(data_tar[0])
        )
    tqdm.write(res)
    RESULT_TEST.append([e, total_loss_test, accuracy])
    log_test.write(res + '\n')


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
    rootdir = 'D:\\GIthub\\RM_TL-main-4-6\\code\\data\\'    
    # rootdir = '/home/xy/wyj/DaNN_wangjindon/office_caltech_10/'
    # rootdir = 'D:\\Laboratory_Work\\A_Transfer_Learning\\office_caltech_10\\'
    torch.manual_seed(1)
    # data_src = data_loader.load_data(
    #     root_dir=rootdir, domain='webcam', batch_size=BATCH_SIZE[0])
    Source = 'office10_surf_amazon'
    Target = 'office10_surf_caltech'

    data1 = data(Source, download=False, root_dir = rootdir)
    train_data, train_data_label = data1.train_data()
    if np.unique(train_data_label)[0] == 1:
        train_data_label -= 1
    # train_data = train_data.reshape(train_data.shape[0], -1)/255.0
    # train_data = train_data/255.0
    data2 = data(Target, download=False, root_dir = rootdir)
    train_data_m, train_data_m_label = data2.train_data()
    if np.unique(train_data_m_label)[0] == 1:
        train_data_m_label -= 1
    # train_data_m = train_data_m.reshape(train_data_m.shape[0], -1)/255.0

    train_data, train_data_label = add_last_batch(train_data, train_data_label)
    source = (train_data, train_data_label)
    train_data_m, train_data_m_label = add_last_batch(train_data_m, train_data_m_label)
    target = (train_data_m, train_data_m_label)


    model = DaNN.DaNN(n_input= FEATURE_NUM, n_hidden=256, n_class=10)
    model = model.to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMEMTUN,
        weight_decay=L2_WEIGHT
    )

    DaNN_time = []    
    for e in tqdm(range(1, N_EPOCH + 1)):
        s_time = time.time()
        model = train(model=model, optimizer=optimizer,
                      epoch=e, data_src=source, data_tar=target)
        DaNN_1epoch_t = time.time() - s_time
        DaNN_time.append(DaNN_1epoch_t)

        test(model, target, e)
    print('DaNN_time=',DaNN_time)
    print('all time=',sum(DaNN_time),'\t average time=',sum(DaNN_time)/len(DaNN_time))
    torch.save(model, 'model_dann.pkl')
    log_train.close()
    log_test.close()
    res_train = np.asarray(RESULT_TRAIN)
    res_test = np.asarray(RESULT_TEST)
    np.savetxt('res_train_a-w.csv', res_train, fmt='%.6f', delimiter=',')
    np.savetxt('res_test_a-w.csv', res_test, fmt='%.6f', delimiter=',')
