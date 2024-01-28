import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import random
from collections import defaultdict
import numpy as np
import torch.utils.data as data
from PIL import Image
import pandas as pd
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        #self.pkl_file = pkl_file
        self.df = pd.read_csv(csv_file)
        self.images = self.df['image_path']
        self.labels = self.df['label']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
    
        label = torch.tensor(self.labels[idx])
        
        return img, label