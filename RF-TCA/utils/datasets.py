from torchvision import datasets
import torch
from torchvision import models, transforms as T
from torch.utils.data.dataloader import DataLoader
from classifier.utils import trainset
import os
import numpy as np
import pickle
import scipy.io as scio
import cv2 as cv

class data():

    def __init__(self, data_name, download=True, root_dir = '/data/code/RM_TL/code/data'):
        
        self.name = data_name
        self.download = download
        self.root_dir = os.path.join(root_dir, data_name)

    def train_data(self):
        if self.name == 'USPS':
            training_data = datasets.USPS(
                root=self.root_dir,
                train=True,
                download=self.download,
            )
            data, label = training_data.data, np.array(training_data.targets)
            data_ = data
            data = []
            for _ in data_:
                d = cv.resize(_, (28, 28))
                data.append(d)
            data = np.array(data)


        elif self.name == 'SVHN':
            training_data = datasets.SVHN(
                root=self.root_dir,
                split='train',
                download=self.download,
            )
            data, label = training_data.data, np.array(training_data.labels)
            data_ = np.transpose(data, (0, 2, 3, 1))
            data = []
            for _ in data_:
                d = cv.resize(_, (28, 28))
                data.append(d)
            data = np.array(data)
            data_ = (data[:,:,:,0] + data[:,:,:,1] + data[:,:,:,2]) / 3
            data = np.array(data_)

        elif self.name == 'MNIST':
            f = open('/data/code/RM_TL/code/data/MNIST/mnist_data.pkl', 'rb')
            data = pickle.load(f)
            f.close()
            data, label = data['train'], data['train_label']
            data_ = []
            for _ in data:
                d = cv.resize(_, (28, 28))
                data_.append(d)
            data_ = np.array(data_)
            data = (data_[:,:,:,0] + data_[:,:,:,1] + data_[:,:,:,2]) / 3
            data = np.array(data)

        elif self.name == 'MNIST-M':
            f = open('/data/code/RM_TL/code/data/MNIST_M/mnist_m_data.pkl', 'rb')
            data = pickle.load(f)
            f.close()
            data, label = data['train'], data['train_label']
            data_ = []
            for _ in data:
                d = cv.resize(_, (28, 28))
                data_.append(d)
            data_ = np.array(data_)
            data = (data_[:,:,:,0] + data_[:,:,:,1] + data_[:,:,:,2]) / 3
            data = np.array(data)

        elif self.name == 'office10_decaf6_amazon':
            dataFile = os.path.join(self.root_dir, 'amazon_decaf.mat')
            data = scio.loadmat(dataFile)
            data, label = data['feas'], data['labels']

        elif self.name == 'office10_decaf6_caltech':
            dataFile = os.path.join(self.root_dir, 'caltech_decaf.mat')
            data = scio.loadmat(dataFile)
            data, label = data['feas'], data['labels']

        elif self.name == 'office10_decaf6_dslr':
            dataFile = os.path.join(self.root_dir, 'dslr_decaf.mat')
            data = scio.loadmat(dataFile)
            data, label = data['feas'], data['labels']

        elif self.name == 'office10_decaf6_webcam':
            dataFile = os.path.join(self.root_dir, 'webcam_decaf.mat')
            data = scio.loadmat(dataFile)
            data, label = data['feas'], data['labels']
        
        elif self.name == 'office10_surf_amazon':
            dataFile = os.path.join(self.root_dir, 'amazon_zscore_SURF_L10.mat')
            data = scio.loadmat(dataFile)
            data, label = data['Xt'], data['Yt']

        elif self.name == 'office10_surf_caltech':
            dataFile = os.path.join(self.root_dir, 'Caltech10_zscore_SURF_L10.mat')
            data = scio.loadmat(dataFile)
            data, label = data['Xt'], data['Yt']

        elif self.name == 'office10_surf_dslr':
            dataFile = os.path.join(self.root_dir, 'dslr_zscore_SURF_L10.mat')
            data = scio.loadmat(dataFile)
            data, label = data['Xt'], data['Yt']

        elif self.name == 'office10_surf_webcam':
            dataFile = os.path.join(self.root_dir, 'webcam_zscore_SURF_L10.mat')
            data = scio.loadmat(dataFile)
            data, label = data['Xt'], data['Yt']

        elif self.name == 'office31_decaf6_amazon':
            dataFile = os.path.join(self.root_dir, 'amazon_fc6.mat')
            data = scio.loadmat(dataFile)
            data, label = data['fts'], data['labels']

        elif self.name == 'office31_decaf6_dslr':
            dataFile = os.path.join(self.root_dir, 'dslr_fc6.mat')
            data = scio.loadmat(dataFile)
            data, label = data['fts'], data['labels']

        elif self.name == 'office31_decaf6_webcam':
            dataFile = os.path.join(self.root_dir, 'webcam_fc6.mat')
            data = scio.loadmat(dataFile)
            data, label = data['fts'], data['labels']

        ###

        elif self.name == 'office31_decaf7_amazon':
            dataFile = os.path.join(self.root_dir, 'amazon_fc7.mat')
            data = scio.loadmat(dataFile)
            data, label = data['fts'], data['labels']

        elif self.name == 'office31_decaf7_dslr':
            dataFile = os.path.join(self.root_dir, 'dslr_fc7.mat')
            data = scio.loadmat(dataFile)
            data, label = data['fts'], data['labels']

        elif self.name == 'office31_decaf7_webcam':
            dataFile = os.path.join(self.root_dir, 'webcam_fc7.mat')
            data = scio.loadmat(dataFile)
            data, label = data['fts'], data['labels']

        else:
            print('There is not the ', self.name, ' dataset.')
        
        return data, label

    def test_data(self):
        if self.name == 'USPS':
            test_data = datasets.USPS(
                root=self.root_dir,
                train=False,
                download=self.download,
            )
            data, label = test_data.data, np.array(test_data.targets)
        elif self.name == 'SVHN':
            test_data = datasets.SVHN(
                root=self.root_dir,
                split='test',
                download=self.download,
            )
            data, label = test_data.data, np.array(test_data.labels)
        elif self.name == 'MNIST':
            f = open('/data/code/RM_TL/code/data/MNIST/mnist_data.pkl', 'rb')
            data = pickle.load(f)
            f.close()
            data, label = data['val'], data['val_label']
        elif self.name == 'MNIST-M':
            f = open('/data/code/RM_TL/code/data/MNIST_M/mnist_m_data.pkl', 'rb')
            data = pickle.load(f)
            f.close()
            data, label = data['val'], data['val_label']
        else:
            print('There is not the ', self.name, ' dataset.')
        
        return data, label


def resnet_18(source_data, source_label):

    transform = T.Compose([T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    data_train_loader = DataLoader(trainset(source_data, source_label, transform=transform), batch_size=500, shuffle=False, num_workers=8)
    resnet18 = models.resnet18(pretrained=True)
    r = np.zeros((1,1))
    l = np.zeros((1,1))
    for i, (images, labels) in enumerate(data_train_loader):
        res = resnet18(images)
        res = res.detach().numpy()
        labels = labels.detach().numpy()
        if r.shape[0] == 1:
            r = res
            l = labels
        else:
            r = np.concatenate((r, res))
            l = np.concatenate((l, labels))
    return r, l

def resnet_34(source_data, source_label):

    transform = T.Compose([T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    data_train_loader = DataLoader(trainset(source_data, source_label, transform=transform), batch_size=500, shuffle=False, num_workers=8)
    resnet34 = models.resnet34(pretrained=True)
    r = np.zeros((1,1))
    l = np.zeros((1,1))
    for i, (images, labels) in enumerate(data_train_loader):
        res = resnet34(images)
        res = res.detach().numpy()
        labels = labels.detach().numpy()
        if r.shape[0] == 1:
            r = res
            l = labels
        else:
            r = np.concatenate((r, res))
            l = np.concatenate((l, labels))
    return r, l