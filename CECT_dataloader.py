import torch
import torchvision
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class CECT_dataset(Dataset):#需要继承data.Dataset
    def __init__(self,path=None):
        self.path = path
        self.data = os.listdir(self.path)



        
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        data_path = os.path.join(self.path,self.data[index])
        data = np.load(data_path)
        label = torch.from_numpy(data['label']).float()
        # label = torch.transpose(label,0,1)
        return torch.from_numpy(data['data'].transpose((3,0,1,2))).float(),label


    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.data)