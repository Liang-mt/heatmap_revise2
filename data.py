import os
import cv2
import numpy as np
import torch

from utils import *
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
from heatmap_label import *
transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        f=open(path)
        self.dataset=f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data=self.dataset[index]
        img_path=data.split(' ')[0]
        image=Image.open(img_path).convert('RGB')
        points=data.split(' ')[1:]
        points=[int(i.rstrip("\n")) for i in points]
        label=[]
        for i in range(0,len(points),2):
            heatmap=CenterLabelHeatMap(128,128,points[i],points[i+1],5) #核的大小可以自己调整
            label.append(heatmap)
        label=np.stack(label)
        return transform(image),torch.Tensor(label)

if __name__ == '__main__':
    data=MyDataset('./datasets/data_center_val.txt')
    print(data[5][0].shape)
    print(data[5][1].shape)
    for i in data:
        print(i[0].shape)