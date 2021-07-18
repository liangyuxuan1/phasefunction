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
        h, w = image.shape
        image = torch.from_numpy(image).reshape(1, h, w)
        image = image.float()
        gt = self.img_labels.iloc[idx, 3]    # 3: g value
        gt = torch.tensor(gt).reshape(-1)
        gt = gt.float()
        return image, gt


training_data = CustomImageDataset(
    annotations_file = "trainDataCW.csv",
    img_dir="imageCW_FixedG_500"
)

# Refer to：https://blog.csdn.net/peacefairy/article/details/108020179

# If the images can be load into memory completely.
train_dataloader = DataLoader(training_data, batch_size=len(training_data))
img = next(iter(train_dataloader))
mean = img[0].mean()
std  = img[0].std()
print(mean)
print(std)

# Results
# imageCW_FixedG_500, mean = 0.0050, std = 0.3737

# ===============================================================
# If there are too many images to load into memory in one batch
train_dataloader = DataLoader(training_data, batch_size=1000)

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

