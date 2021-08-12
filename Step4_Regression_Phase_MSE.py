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

import shutil
import io

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
import scipy.io as scipyIO

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
        image = scipyIO.loadmat(img_path).get('rawData')
        image = image.astype(np.float64)
        h, w = image.shape
        image = torch.from_numpy(image).reshape(1, h, w)
        image = image.float()

        ua = self.img_labels.iloc[idx, 1]    # 1: ua value
        us = self.img_labels.iloc[idx, 2]    # 2: us value
        g = self.img_labels.iloc[idx, 3]     # 3: g value

        gt = torch.tensor([ua, us, g])
        # gt = torch.tensor(gt).reshape(-1)
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
# imageCW, 500*500, 14 materials, training number = 500, mean = 0.0040, sta = 0.4645
# imageCW, 500*500, 12 materials, training number = 500, mean = 0.0047, sta = 0.5010
# gt = [ua, us, g], min = [0.0010, 0.0150, 0.1550], max = [0.2750, 100.92, 0.9550]

# imageCW_v3, 500x500, training number = 80, mean = 0.0026, std = 0.9595

# imageCW_v4, 500x500, training number = 50, mean = 0.0026, std = 0.9595

class gtNormalize(object):
    def __init__(self, minV, maxV):
        self.minV = torch.tensor(minV)
        self.maxV = torch.tensor(maxV)
    
    def __call__(self, gt):
        # normalize gt to [0.01, 1] to facilitate the calculation of relative error
        k = torch.div(1.0-0.01, self.maxV - self.minV)
        gt = 0.01 + k*(gt - self.minV)
        return gt

    def restore(self, gt):
        # restore the normalized values
        k = torch.div(1.0-0.01, self.maxV - self.minV)
        gt = (gt - 0.01)/k + self.minV
        return gt

img_path="imageCW_v3"
train_data = CustomImageDataset(
    annotations_file = os.path.join(img_path, "trainDataCW_v4.csv"),
    img_dir = img_path,
    transform = transforms.Normalize(0.0026, 0.9595),
    target_transform = gtNormalize(minV = [0.0010, 0.01, -1.0], maxV = [10.0, 100.0, 1.0])
)

test_data = CustomImageDataset(
    annotations_file = os.path.join(img_path, "valDataCW_v4.csv"),
    img_dir = img_path,
    transform = transforms.Normalize(0.0026, 0.9595),
    target_transform = gtNormalize(minV = [0.0010, 0.01, -1.0], maxV = [10.0, 100.0, 1.0])
)

gtNorm = gtNormalize(minV = [0.0010, 0.01, -1.0], maxV = [10.0, 100.0, 1.0])

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     figtitle = 'g=%.2f'%label
#     plt.title(figtitle)
#     plt.axis("off")
#     plt.imshow(np.log(np.log(img.squeeze()+1)+1), cmap="hot")
# plt.show()

# We pass the Dataset as an argument to DataLoader. 
# This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading. 
# Here we define a batch size of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels.

# Create data loaders.
batch_size = 120
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

# Creating Models
# To define a neural network in PyTorch, we create a class that inherits from nn.Module. 
# We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function. 
# To accelerate operations in the neural network, we move it to the GPU if available.

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
num_of_Gaussian = 20
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.convLayers = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU()            
        )

        self.fc = nn.Sequential(
            nn.Linear(128*11*11, 3*num_of_Gaussian+2),
            nn.Sigmoid()
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

def kl_divergence(dis_a, dis_b):
    disa = dis_a + 1e-6
    disb = dis_b + 1e-6
    loga = torch.log(disa)
    logb = torch.log(disb)
    part1 = dis_a*loga
    part2 = dis_a*logb
    result = torch.mean(torch.sum(part1-part2, dim=1))
    assert torch.isnan(result).sum() == 0
    return result

def HG_theta(g, theta):
    # calculate 2*pi*p(cos(theta))
    bSize = g.size()[0] 
    p = torch.zeros(bSize, theta.size()[0]).to(device)
    for i in range(bSize):
        p[i,:] = 0.5*(1-g[i]*g[i])/((1+g[i]*g[i]-2*g[i]*torch.cos(theta))**(3.0/2.0) + 1e-6)
        p[i,:] *= torch.sin(theta)
        # print(torch.sum(p[i,:]))
    return p

def normfun(x, mean, sigma):
    pi = np.pi
    pi = torch.tensor(pi)
    std = sigma + 1e-6
    G_x = torch.exp(-((x - mean)**2)/(2*std**2)) / (std * torch.sqrt(2*pi))
    return G_x

def GMM(nnOut, theta):
    pi = torch.tensor(np.pi)
    w = nnOut[:, 0:num_of_Gaussian]                         # weight [0, 1], sum(w)=1
    w_sum = torch.sum(w, dim=1)
    m = nnOut[:, num_of_Gaussian:num_of_Gaussian*2]*pi      # mean [0, 1]*pi
    d = nnOut[:, num_of_Gaussian*2:num_of_Gaussian*3]       # std [0, 1]

    bSize = nnOut.size()[0]
    gmm = torch.zeros(bSize, theta.size()[0]).to(device)
    for i in range(bSize):
        for j in range(num_of_Gaussian):
            gmm[i,:] += (w[i, j]/w_sum[i]) * normfun(theta, m[i, j], d[i, j])
        #print(torch.sum(gmm[i,:]))
    return gmm

# loss_fn = nn.MSELoss()
theta = np.arange(0, np.pi, 0.01)
theta = torch.from_numpy(theta).to(device)

def loss_fn(prediction, gt):
    gmm = GMM(prediction[:, 0:num_of_Gaussian*3], theta)
    mean_sum_GMM = torch.mean(torch.sum(gmm, dim=1))
    
    gx = gtNorm.restore(gt.to("cpu"))
    gt = gt.to(device)
    g = gx[:, 2]
    p_theta = HG_theta(g, theta)

    # loss1 = kl_divergence(gmm, p_theta)
    # loss2 = kl_divergence(p_theta, gmm)
    # loss_phase = (loss1 + loss2)/2.0
    loss_phase = nn.MSELoss()(gmm, p_theta)

    uas = prediction[:, -2:]
    gt_uas = gt[:, :2]
    loss_uas = nn.MSELoss()(uas, gt_uas)  

    loss = loss_phase + loss_uas

    return loss, mean_sum_GMM


# TRICK TWO: use SGDM, stochastic gradient descent with momentum.
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)

# TRICK THREE: use stepwise decreasing learning rate. 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('training_results')

# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
# and backpropagates the prediction error to adjust the model’s parameters.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss, mean_sum_GMM = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>10f}  [{current:>5d}/{size:>5d}], mean sum GMM: {mean_sum_GMM:>5f}")

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
            loss, mean_sum_GMM = loss_fn(pred, y)
            test_loss += loss.item()
            
            pred_uas = pred[:, -2:]
            gt_uas = y[:, :2]
            pred_error = (pred_uas - gt_uas).abs()/gt_uas
            small_error_num = (pred_error <= 0.1).prod(1).sum().item()
            large_error_num = (pred_error >= 0.5).prod(1).sum().item()
            medium_error_num = len(pred_error) - small_error_num - large_error_num
            correct += [small_error_num, medium_error_num, large_error_num]
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct[0]):>0.1f}%, {(100*correct[1]):>0.1f}%, {(100*correct[2]):>0.1f}%, Avg loss: {test_loss:>10f} \n")
    
    return test_loss, correct

# The training process is conducted over several iterations (epochs). 
# During each epoch, the model learns parameters to make better predictions. 
# We print the model’s accuracy and loss at each epoch; we’d like to see the accuracy increase and the loss decrease with every epoch.

# Show the loss curves of training and testing
# https://blog.csdn.net/weixin_42204220/article/details/86352565   does not work properly
# pip install tensorboardX
# from tensorboardX import SummaryWriter

# https://zhuanlan.zhihu.com/p/103630393 , this works
# 不要安装pytorch profiler, 如果安装了，pip uninstall torch-tb-profiler. 否则tensorboard load 数据有问题

# How To Save and Load Model In PyTorch With A Complete Example
# https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
#
# Saving function
def save_ckp(state, checkpoint_path):
    """
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)

# Loading function
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    test_loss_min = checkpoint['test_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], test_loss_min


# show test results and add the figure in writter
# https://tensorflow.google.cn/tensorboard/image_summaries?hl=zh-cn

def show_test_samples():
    cols, rows = 4, 2
    numEachUaGroup = 9*21*10
    sample_idx = np.zeros(cols*rows, dtype=np.int32)
    for i in range (cols*rows):
        sample_idx[i] = np.random.randint(numEachUaGroup) + i*numEachUaGroup
    
    model.eval()

    figure = plt.figure(figsize=(8, 4))
    for i in range(cols * rows):
        idx = sample_idx[i]
        x, gt = test_data[idx]
        x = x.reshape(1,*x.shape)
        gt = gt.reshape(1,-1)
        x, gt = x.to(device), gt.to(device)

        pred = model(x)
        loss, _ = loss_fn(pred, gt)

        gmm = GMM(pred[:, 0:num_of_Gaussian*3], theta)
        gt = gtNorm.restore(gt.to("cpu"))
        g = gt[:, 2]
        g = g.to(device)
        p_theta = HG_theta(g, theta)

        figure.add_subplot(rows, cols, i+1)
        figtitle = 'ua=%.3f, us=%.2f, g=%.1f \n loss=%.4f' %(gt[0, 0], gt[0, 1], gt[0, 2], loss.item())
        plt.title(figtitle)
        plt.axis("on")
        gmm, p_theta = gmm.to("cpu"), p_theta.to("cpu")
        gmm = gmm.detach().numpy()
        p_theta = p_theta.detach().numpy()
        px = theta.to("cpu")
        px = px.detach().numpy()
        plt.plot(px, gmm.squeeze())
        plt.plot(px, p_theta.squeeze())

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # plt.show()
    return figure


# ========================================================
if __name__=='__main__':

    import time
    since = time.time()

    start_epoch = 1
    n_epochs = 30
    test_loss_min = torch.tensor(np.Inf)

    checkpoint_path = 'checkpoints_MSE'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    checkpoint_file = os.path.join(checkpoint_path, 'current_checkpoint.pt')
    best_model_file = os.path.join(checkpoint_path, 'best_model.pt')
    is_best = False

    resume_training = True
    if resume_training:
        model, optimizer, start_epoch, test_loss_min = load_ckp(checkpoint_file, model, optimizer)

    for epoch in range(start_epoch, n_epochs+1):
        print(f"Epoch {epoch}\n-------------------------------")

        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        test_loss, correct = test(test_dataloader, model, loss_fn)

        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Accuracy: relative error < 10%', 100*correct[0], epoch)
        writer.add_scalar('Accuracy: relative error 10-50%', 100*correct[1], epoch)
        writer.add_scalar('Accuracy: relative error > 50%', 100*correct[2], epoch)

        figure = show_test_samples()
        writer.add_figure('Examples of Test Results', figure, epoch)

        if test_loss < test_loss_min:
            print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(test_loss_min, test_loss))
            # save checkpoint as best model
            test_loss_min = test_loss
            is_best = True

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'test_loss_min': test_loss_min,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        # save checkpoint
        save_ckp(checkpoint, checkpoint_file)
        if is_best:
            shutil.copyfile(checkpoint_file, best_model_file)
            is_best = False   

        time_elapsed = time.time() - since
        print('Epoch {:d} complete in {:.0f}m {:.0f}s'.format(epoch, time_elapsed // 60 , time_elapsed % 60))


    writer.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))




