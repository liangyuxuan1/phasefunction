import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
from torchvision.io import read_image

import scipy.io as io


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # 0: filename
        # image = read_image(img_path)
        image = io.loadmat(img_path).get('rawData')
        image = image.astype(np.float64)        
        h, w = image.shape
        image = torch.from_numpy(image).reshape(1, h, w)
        image = image.float()

        # gt = self.img_labels.iloc[idx, 3]    # 3: g value
        # gt = torch.tensor(gt).reshape(-1)
        # gt = gt.float()

        ua = self.img_labels.iloc[idx, 1]    # 1: ua value
        us = self.img_labels.iloc[idx, 2]    # 2: us value
        g = self.img_labels.iloc[idx, 3]     # 3: g value

        gt = torch.tensor([ua, us, g])

        return image, gt


img_path="imageCW_v3"
training_data = CustomImageDataset(
    annotations_file = os.path.join(img_path, "trainDataCW_v4.csv"),
    img_dir = img_path
)

# Refer to：https://blog.csdn.net/peacefairy/article/details/108020179

# If the images can be load into memory completely.

# train_dataloader = DataLoader(training_data, batch_size=len(training_data))
# img, gt = next(iter(train_dataloader))
# mean = img.mean()
# std  = img.std()
# print(mean)
# print(std)

# minGt = gt.min(0)
# maxGt  = gt.max(0)
# print(minGt)
# print(maxGt)

# Results
# imageCW, 500x500, g=0.5:0.01:0.95, training number = 70, mean = 0.0050, std = 0.3737
# imageCW, 500x500, g=-1:0.025:1, training number = 100, mean = 0.0068, std = 1.2836
# imageCW, 500*500, 14 materials, training number = 500, mean = 0.0040, std = 0.4645

# imageCW, 500*500, 12 materials, training number = 500, mean = 0.0047, std = 0.5010
# gt = [ua, us, g], min = [0.0010, 0.0150, 0.1550], max = [0.2750, 100.92, 0.9550]

# imageCW_v3, 500x500, training number = 80, mean = 0.0026, std = 0.9595

# imageCW_v4, 500x500, training number = 50, mean = 0.0026, std = 0.9595

# ===============================================================
# If there are too many images to load into memory in one batch
# train_dataloader = DataLoader(training_data, batch_size=6048) # for v3
train_dataloader = DataLoader(training_data, batch_size=7560)   # for v4

total_sum = 0
for batch in train_dataloader: 
    total_sum += batch[0].sum()
batchSize = batch[0].size()
num_of_pixels = batchSize[2]*batchSize[3]*len(training_data)
mean = total_sum / num_of_pixels
print(mean)

sum_of_squared_error = 0
for batch in train_dataloader: 
    sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
std = torch.sqrt(sum_of_squared_error / num_of_pixels)
print(std)


