import os
import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, opt, transform=None):
        super().__init__()
        self.root = "/data/komedi/data/aflw"
        self.files = sorted(os.listdir(self.root))
        # self.files = sorted(glob.glob(self.root + "/*.*"))
        self.transform = transform
        self.hr_height = opt.hr_height
        
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((self.hr_height // 4, self.hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((self.hr_height, self.hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx]))
        
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
            
        return {"lr": img_lr, "hr": img_hr}