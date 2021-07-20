# PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset. 
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.

import torch
from torch import nn
from torch.nn.modules.activation import Tanh
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose

# pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np
import string

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets. 
# For this tutorial, we will be using a TorchVision dataset.
# The torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR, COCO (full list here). 

# Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. 

# pip install pandas
import pandas as pd
from torchvision.io import read_image

# pip install scipy
import scipy.io as io

#data = io.loadmat(fullFileName, variable_names='rawData', mat_dtype=True)

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # 0: filename
        # image = read_image(img_path)
        image = io.loadmat(img_path).get('rawData')
        image = image.astype(np.float)
        h, w = image.shape
        image = torch.from_numpy(image).reshape(1, h, w)
        image = image.float()

        gt = self.img_labels.iloc[idx, 3]    # 3: g value
        gt = torch.tensor(gt).reshape(-1)
        gt = gt.float()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt = self.target_transform(gt)

        return image, gt

# TRICK ONE to improve the performance: standardize the data.
# Need to calculate the mean and std of the dataset first.
# imageCW, 500x500, g=0.5:0.01:0.95, training number = 70, mean = 0.0050, std = 0.3737
# imageCW, 500x500, g=-1:0.025:1, training number = 100, mean = 0.0068, std = 1.2836
training_data = CustomImageDataset(
    annotations_file = "trainDataCW.csv",
    img_dir="imageCW",
    transform = transforms.Normalize(0.0068, 1.2836)
)

test_data = CustomImageDataset(
    annotations_file = "testDataCW.csv",
    img_dir="imageCW",
    transform = transforms.Normalize(0.0068, 1.2836)
)

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    figtitle = 'g=%.2f'%label
    plt.title(figtitle)
    plt.axis("off")
    plt.imshow(np.log(np.log(img.squeeze()+1)+1), cmap="hot")
plt.show()

# We pass the Dataset as an argument to DataLoader. 
# This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading. 
# Here we define a batch size of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels.

# Create data loaders.
batch_size = 20
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Creating Models
# To define a neural network in PyTorch, we create a class that inherits from nn.Module. 
# We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function. 
# To accelerate operations in the neural network, we move it to the GPU if available.

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.convLayers = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.flatten(x)
        x = self.convLayers(x)
        x = x.view(x.size(0), -1)
        pred = self.fc(x)

        return pred

model = NeuralNetwork().to(device)
# print(model)
# pip install torchsummary
from torchsummary import summary
summary(model, (1, 500, 500))

# Optimizing the Model Parameters
# To train a model, we need a loss function and an optimizer.
loss_fn = nn.MSELoss()

# TRICK TWO: use SGDM, stochastic gradient descent with momentum.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# TRICK THREE: use stepwise decreasing learning rate. 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
# and backpropagates the prediction error to adjust the model’s parameters.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>10f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    scheduler.step()

    return train_loss

# We also check the model’s performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, np.zeros(3)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_error = (pred - y).abs()
            small_error_num = (pred_error < 0.001).sum().item()
            large_error_num = (pred_error >= 0.002).sum().item()
            medium_error_num = len(pred_error) - small_error_num - large_error_num
            correct += [small_error_num, medium_error_num, large_error_num]
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct[0]):>0.1f}%, {(100*correct[1]):>0.1f}%, {(100*correct[2]):>0.1f}%, Avg loss: {test_loss:>10f} \n")

    return test_loss

# The training process is conducted over several iterations (epochs). 
# During each epoch, the model learns parameters to make better predictions. 
# We print the model’s accuracy and loss at each epoch; we’d like to see the accuracy increase and the loss decrease with every epoch.

# Show the loss curves of training and testing
# https://blog.csdn.net/weixin_42204220/article/details/86352565   does not work properly
# pip install tensorboardX
# from tensorboardX import SummaryWriter

# https://zhuanlan.zhihu.com/p/103630393 , this works
# 不要安装pytorch profiler, 如果安装了，pip uninstall torch-tb-profiler. 否则tensorboard load 数据有问题
from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('training_results')

import time
since = time.time()

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    test_loss  = test(test_dataloader, model, loss_fn)

    writer.add_scalar('Train/Loss', train_loss, t)
    writer.add_scalar('Test/Loss', test_loss, t)
writer.close()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))

# Saving Models
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# Loading Models
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))



