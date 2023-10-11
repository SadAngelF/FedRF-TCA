import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from .utils import trainset
from torchvision import transforms as T


class MNN(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self, input_dim):
        super(MNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.Softmaxmax = nn.LogSoftmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, img):
        output = self.relu(self.fc1(img))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        output = self.Softmaxmax(output)
        return output


class MNN_classifier():

    def __init__(self, input_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MNN(input_dim).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=2e-3)

    def train(self, data, label, epoch):
        label = label.reshape(-1)
        c = len(np.unique(label))
        if np.unique(label)[0] == 1:
            label -= 1
        label = np.eye(c)[label]
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float() 
        data_train_loader = DataLoader(trainset(data, label), batch_size=100, shuffle=True, num_workers=1)
        self.net.train()
        loss_list, batch_list = [], []
        for i, (images, labels) in enumerate(data_train_loader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.net(images)
            loss = self.criterion(output, labels)

            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i+1)

            # if i % 10 == 0:
            #     print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

            loss.backward()
            self.optimizer.step()

    def test(self, data, label, Epoch):
        self.net.eval()
        label = label.reshape(-1)
        c = len(np.unique(label))
        if np.unique(label)[0] == 1:
            label -= 1
        label = np.eye(c)[label]
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()
        data_test_loader = DataLoader(trainset(data, label), batch_size=100, num_workers=1)
        total_correct = 0
        avg_loss = 0.0
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.net(images)
            avg_loss += self.criterion(output, labels).sum()
            pred = output.detach().cpu().argmax(1)
            y = labels.detach().cpu().argmax(1)
            total_correct += (pred == y).float().sum().item()

        avg_loss /= len(data)
        print('Test - Epoch %d, Avg. Loss: %f, Accuracy: %f' % (Epoch, avg_loss.detach().cpu().item(), float(total_correct) / len(data)))
        return float(total_correct) / len(data)

