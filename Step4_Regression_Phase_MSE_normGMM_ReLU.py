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
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Agg")
import numpy as np
import time
import shutil

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets. 
# For this tutorial, we will be using a TorchVision dataset.
# The torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR, COCO (full list here). 

# Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. 

# pip install torchsummary
from torchsummary import summary
# pip install openpyxl
import openpyxl

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
        img_label = self.img_labels.iloc[idx, :]
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

        return image, gt, img_label

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

# Creating Models
# To define a neural network in PyTorch, we create a class that inherits from nn.Module. 
# We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function. 
# To accelerate operations in the neural network, we move it to the GPU if available.
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
        sumGmm = torch.sum(gmm[i,:]) * 0.01     # discretization bin = 0.01 radian
        gmm[i,:] /= sumGmm      # normalize to gurrantee the sum=1
    return gmm

def loss_fn(prediction, gt):
    gmm = GMM(prediction[:, 0:num_of_Gaussian*3], theta)
    
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

    return loss

# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
# and backpropagates the prediction error to adjust the model’s parameters.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    current = 0
    for batch, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current += len(X)
        if (batch+1) % 10 == 0:
            print(f"loss: {loss.item():>0.6f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    print(f"loss: {train_loss:>10f}  [{current:>5d}/{size:>5d}]")

    scheduler.step()

    return train_loss

# We also check the model’s performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, np.zeros(3)
    with torch.no_grad():
        for X, y, _ in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
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

def show_result_samples(dataset, showFig=False):
    cols, rows = 4, 2
    numEachUaGroup = len(dataset)/(cols*rows)
    sample_idx = np.zeros(cols*rows, dtype=np.int32)
    for i in range (cols*rows):
       sample_idx[i] = np.random.randint(numEachUaGroup) + i*numEachUaGroup
    
    model.eval()

    figure = plt.figure(figsize=(20, 10))
    for i in range(cols * rows):
        idx = sample_idx[i]
        x, gt, label = dataset[idx]
        x = x.reshape(1,*x.shape)
        gt = gt.reshape(1,-1)
        x, gt = x.to(device), gt.to(device)

        pred = model(x)
        loss = loss_fn(pred, gt)

        pred = pred.detach()

        gmm = GMM(pred[:, 0:num_of_Gaussian*3], theta)
        # gt = gtNorm.restore(gt.to("cpu"))
        g = label[:, 2]
        g = g.to(device)
        p_theta = HG_theta(g, theta)

        figure.add_subplot(rows, cols, i+1)
        figtitle = 'ua=%.3f, us=%.2f, g=%.2f \n loss=%.4f' %(gt[0, 0], gt[0, 1], gt[0, 2], loss.item())
        plt.title(figtitle)
        plt.axis("on")
        gmm, p_theta = gmm.to("cpu"), p_theta.to("cpu")
        gmm = gmm.numpy()
        p_theta = p_theta.numpy()
        px = theta.to("cpu")
        px = px.numpy()
        plt.plot(px, gmm.squeeze())
        plt.plot(px, p_theta.squeeze())

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    if showFig:
        plt.show()
    return figure

def show_Results(dataset, figure_path, save_figure=False):
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    model.eval()

    results = np.zeros((len(dataset), 9))   # ['ua', 'pred_ua', 'ua_re', 'us', 'pred_us', 'us_re', 'g', 'phase_mse', 'numPixels']
    for i in range(len(dataset)):
        x, _, label = dataset[i]
        img = x*stdPixelVal + meanPixelVal
        x = x.reshape(1,*x.shape)
        x = x.to(device)
        pred = model(x)
        # loss = loss_fn(pred, gt)

        gmm = GMM(pred[:, 0:num_of_Gaussian*3], theta)
        g = label[3]
        g = g.reshape(1,-1)
        g = torch.from_numpy(g)
        g = g.to(device)
        p_theta = HG_theta(g, theta)
        phase_mse = nn.MSELoss()(gmm, p_theta)
        phase_mse = phase_mse.detach()
        phase_mse = phase_mse.to("cpu")

        pred = pred.detach()
        pred = pred.squeeze()
        pred = pred.to("cpu")

        ua, us = label[1], label[2]
        tmp = torch.tensor([pred[-2], pred[-1], np.random.rand()])
        pred_rst = gtNorm.restore(tmp)      # change the network output to parameter range
        pred_ua, pred_us = pred_rst[0], pred_rst[1]
        ua_re, us_re = np.abs(ua-pred_ua)/ua, np.abs(us-pred_us)/us

        results[i,:] = [ua, pred_ua, ua_re, us, pred_us, us_re, label[3], phase_mse, label[4]]

        if save_figure and (i % 10 == 0):
            fig = plt.figure(figsize=(8, 4))
            plt.axis("off")
            figtitle = 'ua=%.3f, us=%.2f, g=%.2f, Phase MSE=%.4f \n' %(ua, us, g, phase_mse)
            plt.title(figtitle)

            fig.add_subplot(1, 2, 1)
            plt.axis("off")
            img = np.float_power(img.squeeze(), 0.1)
            plt.imshow((img), cmap="hot")

            fig.add_subplot(1, 2, 2)
            plt.axis("on")
            gmm, p_theta = gmm.detach(),  p_theta.detach()
            gmm, p_theta = gmm.to("cpu"), p_theta.to("cpu")
            gmm, p_theta = gmm.numpy(),   p_theta.numpy()
            px = theta.to("cpu")
            px = px.numpy()
            plt.plot(px, gmm.squeeze(), label='est')
            plt.plot(px, p_theta.squeeze(), label='gt')
            plt.legend()
            
            figFileName = 'Fig_%04d.png' % (i/10 + 1)
            figFile = os.path.join(figure_path, figFileName)
            plt.savefig(figFile)
            plt.close('all')
       
    return results

def write_results_exel(results, filename):
    data_df = pd.DataFrame(results)
    data_df.columns = ['ua', 'pred_ua', 'ua_RE', 'us', 'pred_us', 'us_RE', 'g', 'phase_MSE', 'numPixels']
    # need to install openpyxl: pip install openpyxl
    # and import openpyxl
    tb_writer = pd.ExcelWriter(filename)
    data_df.to_excel(tb_writer, 'page_1', float_format='%.4f')
    tb_writer.save()
    tb_writer.close()

# ==============================================================================================================
if __name__=='__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Need to calculate the mean and std of the dataset first.

    # imageCW, 500x500, g=0.5:0.01:0.95, training number = 70, mean = 0.0050, std = 0.3737
    # imageCW, 500x500, g=-1:0.025:1, training number = 100, mean = 0.0068, std = 1.2836
    # imageCW, 500*500, 14 materials, training number = 500, mean = 0.0040, sta = 0.4645
    # imageCW, 500*500, 12 materials, training number = 500, mean = 0.0047, sta = 0.5010
    # gt = [ua, us, g], min = [0.0010, 0.0150, 0.1550], max = [0.2750, 100.92, 0.9550]

    # imageCW_v3, 500x500, training number = 80, mean = 0.0026, std = 0.9595
    # imageCW_v4, 500x500, training number = 50, mean = 0.0026, std = 0.9595
    # trainDataCW_v3_ExcludeExtremes, 500x500, training number = 80, mean = 0.0028, std = 0.8302

    img_path = "H:/imageCW_v3"
    test_img_path = "imageCW_test"
    trainDataListFile = "trainDataCW_v3_ExcludeExtremes_small.csv"
    valDataListFile   = "valDataCW_v3_ExcludeExtremes_small.csv"
    testDataListFile  = "testDataCW_ExcludeExtremes.csv"
    meanPixelVal = 0.0028
    stdPixelVal  = 0.8302
    minParaVal   = [0.0010, 0.01, -0.9]
    maxParaVal   = [10.0, 100.0, 0.9]

    train_data = CustomImageDataset(
        annotations_file = os.path.join(img_path, trainDataListFile),
        img_dir = img_path,
        transform = transforms.Normalize(meanPixelVal, stdPixelVal),
        target_transform = gtNormalize(minParaVal, maxParaVal)
    )

    val_data = CustomImageDataset(
        annotations_file = os.path.join(img_path, valDataListFile),
        img_dir = img_path,
        transform = transforms.Normalize(meanPixelVal, stdPixelVal),
        target_transform = gtNormalize(minParaVal, maxParaVal)
    )

    test_data = CustomImageDataset(
        annotations_file = os.path.join(test_img_path, testDataListFile),
        img_dir = test_img_path,
        transform = transforms.Normalize(meanPixelVal, stdPixelVal),
        target_transform = gtNormalize(minParaVal, maxParaVal)
    )

    gtNorm = gtNormalize(minParaVal, maxParaVal)

    # Create data loaders.
    batch_size = 120
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader   = DataLoader(val_data, batch_size=batch_size, pin_memory=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

    # Define model
    num_of_Gaussian = 10
    model = NeuralNetwork().to(device)
    # print(model)
    summary(model, (1, 500, 500))

    # loss_fn = nn.MSELoss()
    theta = np.arange(0, np.pi, 0.01)
    theta = torch.from_numpy(theta).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_epoch = 31
    n_epochs = 30
    test_loss_min = torch.tensor(np.Inf)

    checkpoint_path = 'training_results'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    checkpoint_file = os.path.join(checkpoint_path, 'current_checkpoint.pt')
    best_model_file = os.path.join(checkpoint_path, 'best_model.pt')
    is_best = False

    resume_training = False
    if resume_training:
        model, optimizer, start_epoch, test_loss_min = load_ckp(checkpoint_file, model, optimizer)

    from torch.utils.tensorboard import SummaryWriter 
    writer = SummaryWriter(checkpoint_path)

    since = time.time()
    for epoch in range(start_epoch, n_epochs+1):
        print(f"Epoch {epoch}")

        train_loss = train(train_dataloader, model, loss_fn, optimizer)
        test_loss, correct = test(val_dataloader, model, loss_fn)

        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Accuracy: relative error < 10%', 100*correct[0], epoch)
        writer.add_scalar('Accuracy: relative error 10-50%', 100*correct[1], epoch)
        writer.add_scalar('Accuracy: relative error > 50%', 100*correct[2], epoch)

        figure = show_result_samples(val_data)
        writer.add_figure('Examples of Validation Results', figure, epoch)

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

    # finally, show results of best model
    model, optimizer, start_epoch, test_loss_min = load_ckp(best_model_file, model, optimizer)
    
    # ----- Validation Results --------------------------------------------------------------------
    
    # ***** ATTENSION: must uncomment matplotlib.use("Agg") to save figures *****
    
    val_results  = show_Results(val_data,  'validating_results_figures', save_figure=False)
    fn_val = os.path.join('validating_results_figures', 'val_results.npy')
    np.save(fn_val, val_results)

    # xlsx_val = os.path.join('validating_results_figures', 'val_results.xlsx')
    # write_results_exel(val_results, xlsx_val)

    # ----- Test Results ---------------------------------------------------------------------------
    test_results = show_Results(test_data, 'testing_results_figures', save_figure=False)
    fn_test = os.path.join('testing_results_figures', 'test_results.npy')
    np.save(fn_test, test_results)

    # xlsx_test = os.path.join('testing_results_figures', 'test_results.xlsx')
    # write_results_exel(test_results, xlsx_test)

    # ----- Compare Results -------------------------------------------------------------------------
    val_results  = np.load(fn_val)
    test_results = np.load(fn_test)
    
    # show estimation of ua
    # compress the dynamic range of relative errors
    val_ua_re = np.power(val_results[:,2], 0.3)
    test_ua_re = np.power(test_results[:,2], 0.3)

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(2, 1, 1)
    plt.title("Validation")
    plt.plot(val_ua_re, label='ua_RE^0.3', color='tab:green')
    plt.plot(val_results[:,1], label='pred_ua', color='tab:blue')
    plt.plot(val_results[:,0], label='ua', color='tab:orange')
    plt.legend()
    plt.axis("on")
    fig.add_subplot(2, 1, 2)
    plt.title("Test")
    plt.plot(test_ua_re, label='ua_RE^0.3', color='tab:green')
    plt.plot(test_results[:,1], label='pred_ua', color='tab:blue')
    plt.plot(test_results[:,0], label='ua', color='tab:orange')
    plt.xlabel('\n Images (sorted in ascending order by ua)')
    plt.legend()
    plt.axis("on")
    plt.show()

    # show estimation of us
    val_results = val_results[np.argsort(val_results[:, 3])]    # sorting results according to us 
    test_results = test_results[np.argsort(test_results[:, 3])]

    val_us_re = np.power(val_results[:,5], 0.6)
    test_us_re = np.power(test_results[:,5], 0.6)

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(2, 1, 1)
    plt.title("Validation")
    plt.plot(val_us_re, label='us_RE^0.6', color='tab:green')
    plt.plot(val_results[:,4], label='pred_us', color='tab:blue')
    plt.plot(val_results[:,3], label='us', color='tab:orange')
    plt.legend()
    plt.axis("on")
    fig.add_subplot(2, 1, 2)
    plt.title("Test")
    plt.plot(test_us_re, label='us_RE^0.6', color='tab:green')
    plt.plot(test_results[:,4], label='pred_us', color='tab:blue')
    plt.plot(test_results[:,3], label='us', color='tab:orange')
    plt.xlabel('\n Images (sorted in ascending order by us)')
    plt.legend()
    plt.axis("on")
    plt.show()

    # show mse of phase function, sorting by g
    val_results = val_results[np.argsort(val_results[:, 6])]    # sorting results according to g 
    test_results = test_results[np.argsort(test_results[:, 6])]

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(2, 1, 1)
    plt.title("Validation")
    plt.plot(val_results[:,7], label='phase_MSE')
    plt.plot(val_results[:,6], label='g')
    plt.legend()
    plt.axis("on")
    fig.add_subplot(2, 1, 2)
    plt.title("Test")
    plt.plot(test_results[:,7], label='phase_MSE')
    plt.plot(test_results[:,6], label='g')
    plt.xlabel('\n Images (sorted in ascending order by g)')
    plt.legend()
    plt.axis("on")
    plt.show()

    # show mse of phase function, sorting by number of non-zeros pixels
    val_results = val_results[np.argsort(val_results[:, 8])]    # sorting results according to num of pixels 
    test_results = test_results[np.argsort(test_results[:, 8])]

    fig = plt.figure(figsize=(10, 8))
    plt.title("Validation")
    plt.axis("off")
    fig.add_subplot(3, 1, 1)
    plt.plot(val_results[:,7], label='phase_MSE')
    plt.legend()
    plt.axis("on")
    fig.add_subplot(3, 1, 2)
    plt.plot(val_results[:,2], label='ua_RE')
    plt.legend()
    plt.axis("on")
    fig.add_subplot(3, 1, 3)
    plt.plot(val_results[:,5], label='us_RE')
    plt.xlabel('\n Images (sorted in ascending order by number of non-zero pixels)')
    plt.legend()
    plt.axis("on")

    fig = plt.figure(figsize=(10, 8))
    plt.title("Test")
    plt.axis("off")
    fig.add_subplot(3, 1, 1)
    plt.plot(test_results[:,7], label='phase_MSE')
    plt.legend()
    plt.axis("on")
    fig.add_subplot(3, 1, 2)
    plt.plot(test_results[:,2], label='ua_RE')
    plt.legend()
    plt.axis("on")
    fig.add_subplot(3, 1, 3)
    plt.plot(test_results[:,5], label='us_RE')
    plt.xlabel('\n Images (sorted in ascending order by number of non-zero pixels)')
    plt.legend()
    plt.axis("on")
    plt.show()
    # --------------------------------------------------------------------------------------------
    show_result_samples(val_data, showFig=True)
    # show_result_samples(test_data, showFig=True)