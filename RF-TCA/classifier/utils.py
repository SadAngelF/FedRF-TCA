from torch._C import dtype
from torch.utils.data import Dataset, DataLoader
import torch

class trainset(Dataset):

    def __init__(self, data, label, transform = None):
        
        self.len = len(data)
        self.images = data # torch.from_numpy(data).float()
        self.target = label # torch.from_numpy(label).float()
        self.transform = transform
       
    def __getitem__(self, index):
        if index > self.len - 1:
            index = index % self.len
        img = self.images[index]
        if self.transform:
            img = self.transform(img)
        target = self.target[index]
        return img,target

    def __len__(self):
        return self.len