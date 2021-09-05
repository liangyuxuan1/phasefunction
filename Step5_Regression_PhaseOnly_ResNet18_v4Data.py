# PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset. 
# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.

from numpy.core.fromnumeric import mean
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter 

# pip install torch-summary
from torchsummary import summary

import logging

# pip install matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Agg")
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import seaborn as sns
import numpy as np
import shutil

# pip install pandas
import pandas as pd

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.append('./src')
from preprocessing import DataPreprocessor
from CustomImageDataset_Pickle import CustomImageDataset_Pickle
from CustomImageDataset import CustomImageDataset
import checkpoints
from logger import double_logger
import trainer
import tester

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
    p = torch.zeros(bSize, theta.size()[0]).cuda()
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
    gmm = torch.zeros(bSize, theta.size()[0]).cuda()
    for i in range(bSize):
        for j in range(num_of_Gaussian):
            gmm[i,:] += (w[i, j]/w_sum[i]) * normfun(theta, m[i, j], d[i, j])
        sumGmm = torch.sum(gmm[i,:]) * 0.01     # discretization bin = 0.01 radian
        gmm[i,:] /= sumGmm      # normalize to gurrantee the sum=1
    return gmm

def loss_func_mse(prediction, gt):
    gmm = GMM(prediction, theta)
    
    # gx = gtNorm.restore(gt.to("cpu"))
    # gt = gt.to(device)
    # g = gx[:, 2]
    g = gt[:,2]
    p_theta = HG_theta(g, theta)

    # loss1 = kl_divergence(gmm, p_theta)
    # loss2 = kl_divergence(p_theta, gmm)
    # loss_phase = (loss1 + loss2)/2.0
    
    loss_phase = nn.MSELoss()(gmm, p_theta)

    # loss_phase = (kl_divergence(gmm, p_theta) + kl_divergence(p_theta, gmm))/2.0

    # uas = prediction[:, -2:]
    # gt_uas = gt[:, :2]
    # loss_uas = nn.MSELoss()(uas, gt_uas)  

    #loss = loss_phase + loss_uas

    loss = loss_phase

    return loss

# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
# and backpropagates the prediction error to adjust the model’s parameters.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    current = 0
    for batch, (X, y) in enumerate(dataloader):
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
    print(f"loss: {train_loss:>0.6f}  [{current:>5d}/{size:>5d}]")

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
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            
            # pred_uas = pred[:, -2:]
            # gt_uas = y[:, :2]
            # pred_error = (pred_uas - gt_uas).abs()/gt_uas
            # small_error_num = (pred_error <= 0.1).prod(1).sum().item()
            # large_error_num = (pred_error >= 0.5).prod(1).sum().item()
            # medium_error_num = len(pred_error) - small_error_num - large_error_num
            # correct += [small_error_num, medium_error_num, large_error_num]
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct[0]):>0.1f}%, {(100*correct[1]):>0.1f}%, {(100*correct[2]):>0.1f}%, Avg loss: {test_loss:>10f} \n")
    print(f"Validation Avg loss: {test_loss:>0.6f}")

    return test_loss, correct

# show test results and add the figure in writter
# https://tensorflow.google.cn/tensorboard/image_summaries?hl=zh-cn

def show_result_samples(dataset, showFig=False):
    cols, rows = 4, 2
    numEachUaGroup = len(dataset)/(cols*rows)
    sample_idx = np.zeros(cols*rows, dtype=np.int32)
    for i in range (cols*rows):
       sample_idx[i] = np.random.randint(numEachUaGroup) + i*numEachUaGroup
    
    model.eval()

    figure = plt.figure(figsize=(18, 9))
    label = dataset.img_labels
    for i in range(cols * rows):
        idx = sample_idx[i]
        x, gt = dataset[idx]
        x = x.reshape(1,*x.shape)
        gt = gt.reshape(1,-1)
        x, gt = x.to(device), gt.to(device)

        pred = model(x)
        loss = loss_fn(pred, gt)

        pred = pred.detach()

        gmm = GMM(pred[:, 0:num_of_Gaussian*3], theta)
        # gt = gtNorm.restore(gt.to("cpu"))
        g = label.iloc[idx, 3]
        g = torch.tensor([g])
        g = g.to(device)
        p_theta = HG_theta(g, theta)

        ua = label.iloc[idx, 1]
        us = label.iloc[idx, 2]

        figure.add_subplot(rows, cols, i+1)
        figtitle = 'ua=%.3f, us=%.2f, g=%.2f \n loss=%.4f' %(ua, us, g, loss.item())
        plt.title(figtitle)
        plt.axis("on")
        gmm, p_theta = gmm.to("cpu"), p_theta.to("cpu")
        gmm = gmm.numpy()
        p_theta = p_theta.numpy()
        px = theta.to("cpu")
        px = px.numpy()
        plt.plot(px, gmm.squeeze())
        plt.plot(px, p_theta.squeeze())

    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    if showFig:
        plt.show()
    return figure

def show_Results(dataset, figure_path, save_figure=False):
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    model.eval()

    results = np.zeros((len(dataset), 9))   # ['ua', 'pred_ua', 'ua_re', 'us', 'pred_us', 'us_re', 'g', 'phase_mse', 'numPixels']
    label = dataset.img_labels
    for i in range(len(dataset)):
        x, _ = dataset[i]
        img = x*stdPixelVal + meanPixelVal
        x = x.reshape(1,*x.shape)
        x = x.to(device)
        pred = model(x)
        # loss = loss_fn(pred, gt)

        gmm = GMM(pred[:, 0:num_of_Gaussian*3], theta)
        g = label.iloc[i, 3]
        g = torch.tensor([g])
        g = g.to(device)
        p_theta = HG_theta(g, theta)
        phase_mse = nn.MSELoss()(gmm, p_theta)
        phase_mse = phase_mse.detach()
        phase_mse = phase_mse.to("cpu")

        pred = pred.detach()
        pred = pred.squeeze()
        pred = pred.to("cpu")

        ua, us = label.iloc[i, 1], label.iloc[i, 2]
        # tmp = torch.tensor([pred[-2], pred[-1], np.random.rand()])
        # pred_rst = gtNorm.restore(tmp)      # change the network output to parameter range
        # pred_ua, pred_us = pred_rst[0], pred_rst[1]
        # ua_re, us_re = np.abs(ua-pred_ua)/ua, np.abs(us-pred_us)/us
        pred_ua, pred_us = 0, 0
        ua_re, us_re = 0, 0

        results[i,:] = [ua, pred_ua, ua_re, us, pred_us, us_re, label.iloc[i, 3], phase_mse, label.iloc[i, 4]]

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
            plt.savefig(figFile, bbox_inches='tight')
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

def write_results_txt(results, filename):
    fid = open(filename, 'w')
    for i in range(np.size(results, 0)):
        for j in range(np.size(results, 1)):
            fid.write('%04f\t' % results[i,j])
        fid.write('%04f\n' % np.mean(results[i,:]))
    fid.write('\n%04f\n\n' % np.mean(results))

    model_struct = summary(model, (1, 500, 500), verbose=0)
    model_struct_str = str(model_struct)
    fid.write(model_struct_str)

    fid.close()

# ==============================================================================================================
if __name__=='__main__':

    # Need to calculate the mean and std of the dataset first.

    # imageCW, 500x500, g=0.5:0.01:0.95, training number = 70, mean = 0.0050, std = 0.3737
    # imageCW, 500x500, g=-1:0.025:1, training number = 100, mean = 0.0068, std = 1.2836
    # imageCW, 500*500, 14 materials, training number = 500, mean = 0.0040, sta = 0.4645
    # imageCW, 500*500, 12 materials, training number = 500, mean = 0.0047, sta = 0.5010
    # gt = [ua, us, g], min = [0.0010, 0.0150, 0.1550], max = [0.2750, 100.92, 0.9550]

    # imageCW_v3, 500x500, training number = 80, mean = 0.0026, std = 0.9595
    # imageCW_v4, 500x500, training number = 50, mean = 0.0026, std = 0.9595
    # trainDataCW_v3_ExcludeExtremes, 500x500, training number = 80, mean = 0.0028, std = 0.8302

    # imageCW_v4, 500x500, training number = 200, mean = 0.0045, std = 0.3633
    # imageCW_v4_fat, 500x500, training number = 200, mean = 0.0068, std = 0.3823

    img_path = "H:\\imageCW_v4"
    test_img_path = "H:\\imageCW_v4_test"
    trainDataListFile = "trainDataCW_v4.csv"
    valDataListFile   = "valDataCW_v4.csv"
    testDataListFile  = "testDataCW_v4.csv"
    tmp_processed_data_dir = "H:\\temp_processed_data"
    checkpoint_path = 'training_results'

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    logger = double_logger(log_path=checkpoint_path).getLogger()

    train_labels = pd.read_csv(os.path.join(img_path, trainDataListFile))
    val_labels   = pd.read_csv(os.path.join(img_path, valDataListFile))
    test_labels  = pd.read_csv(os.path.join(test_img_path, testDataListFile))

    meanPixelVal = 0.0045   # using statistics of all v4 data
    stdPixelVal  = 0.3633
    # minParaVal   = [0.00504, 20.45447, 0.55]    # gt does not need normalization
    # maxParaVal   = [0.00504, 20.45447, 0.95]
    preprocessing_transformer = transforms.Normalize(meanPixelVal, stdPixelVal)
    inverse_preprocessing_transformer = transforms.Normalize(-meanPixelVal, 1.0/stdPixelVal)

    train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomVerticalFlip(0.5)
    ])

    temp_train_pickle_file_name = 'train.pkl'
    temp_val_pickle_file_name   = 'val.pkl'
    temp_test_pickle_file_name  = 'test.pkl'

    need_preprocessing = False
    if need_preprocessing:
        if os.path.exists(tmp_processed_data_dir):
            shutil.rmtree(tmp_processed_data_dir) 
        os.makedirs(tmp_processed_data_dir, exist_ok=False)

        logger.debug('Preprocessing... Saving to {}'.format(tmp_processed_data_dir))
        DataPreprocessor().dump(train_labels, img_path, tmp_processed_data_dir, temp_train_pickle_file_name, preprocessing_transformer)
        DataPreprocessor().dump(val_labels,   img_path, tmp_processed_data_dir, temp_val_pickle_file_name,   preprocessing_transformer)
        DataPreprocessor().dump(test_labels,  img_path, tmp_processed_data_dir, temp_test_pickle_file_name,  preprocessing_transformer)

    train_data = CustomImageDataset_Pickle(
        img_labels = train_labels,
        file_preprocessed = os.path.join(tmp_processed_data_dir, temp_train_pickle_file_name),
        transform = train_transformer
    )

    val_data = CustomImageDataset_Pickle(
        img_labels = val_labels,
        file_preprocessed = os.path.join(tmp_processed_data_dir, temp_val_pickle_file_name)
    )

    test_data = CustomImageDataset_Pickle(
        img_labels = test_labels,
        file_preprocessed = os.path.join(tmp_processed_data_dir, temp_test_pickle_file_name)
    )

    # Create data loaders.
    batch_size = 120
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader   = DataLoader(val_data,   batch_size=batch_size, pin_memory=True)
    test_dataloader  = DataLoader(test_data,  batch_size=batch_size, pin_memory=True)

    torch.backends.cudnn.benchmark = True

    # Define model
    num_of_Gaussian = 7
    # model = NeuralNetwork().to(device)
    # print(model)
    # from resnet_models import resnet14
    # from resnet import resnet18
    from NetworkModels import Resnet18
    model = Resnet18(num_classes=num_of_Gaussian*3)
    model_struct = summary(model, (1, 500, 500), verbose=0)
    model_struct_str = str(model_struct)
    logger.info('Model structure:\n {}'.format(model_struct_str))

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    theta = np.arange(0, np.pi, 0.01)
    theta = torch.from_numpy(theta).cuda()

    need_train = False
    if need_train:
        Trn = trainer.Trainer()
        bestmodel_name = 'best_model_1' 
        logger.info(f'Training {bestmodel_name}\n')
        df_loss = Trn.run(train_dataloader, val_dataloader, model, loss_func_mse, optimizer, scheduler, num_epochs=30, 
                                model_dir=checkpoint_path, model_name=bestmodel_name)
        df_loss.to_csv(os.path.join(checkpoint_path, bestmodel_name+'_loss.csv'), index=False)

        plt.subplots(figsize=(8,4))
        sns.lineplot(data=df_loss, x='Epoch', y='Error', hue='Events')
        figFile = os.path.join(checkpoint_path, f'{bestmodel_name}_loss.png')
        plt.savefig(figFile, bbox_inches='tight')
        plt.close()

    tst = tester.Tester()
    test_results = tst.run(test_data, model, loss_func_mse, checkpoint_path, 'best_model_1.pt', inverse_preprocessing_transformer)
    test_results.to_csv(os.path.join(checkpoint_path, bestmodel_name+'_test_results.csv'), index=False)

    # =============================== Test model ======================================

    model, _, _, _ = checkpoints.load_ckp(best_model_file, model, optimizer)

    # ***** ATTENSION: must uncomment matplotlib.use("Agg") to save figures *****
    val_results  = show_Results(val_data,  os.path.join(checkpoint_path, 'val_results_figures'), save_figure=True)
    fn_val = os.path.join(checkpoint_path, 'val_results.npy')
    np.save(fn_val, val_results)

    # xlsx_val = os.path.join(checkpoint_path, 'val_results.xlsx')
    # write_results_exel(val_results, xlsx_val)

    # ----- Test Results ---------------------------------------------------------------------------
    test_results = show_Results(test_data, os.path.join(checkpoint_path, 'test_results_figures'), save_figure=True)
    fn_test = os.path.join(checkpoint_path, 'test_results.npy')
    np.save(fn_test, test_results)

    # xlsx_test = os.path.join(checkpoint_path, 'test_results.xlsx')
    # write_results_exel(test_results, xlsx_test)

    # ----- Compare Results -------------------------------------------------------------------------
    val_results  = np.load(fn_val)
    test_results = np.load(fn_test)
    
    pNum = 6        # 6 kinds of tissues
    gValNum = 5     # number of g values for val
    gTestNum = 4    # number of g values for test
    iNum = 30       # number of images each parameter set
    val_mean    = np.zeros((pNum, gValNum))
    test_mean   = np.zeros((pNum, gTestNum))
    for i in range(pNum):
        val_pi = val_results[i*iNum*gValNum:(i+1)*iNum*gValNum,:]
        for j in range(gValNum):
            val_pi_gi = val_pi[j*iNum:(j+1)*iNum, :]
            val_mean[i,j] = np.mean(val_pi_gi[:,7])

        test_pi = test_results[i*iNum*gTestNum:(i+1)*iNum*gTestNum,:]
        for j in range(gTestNum):
            test_pi_gi = test_pi[j*iNum:(j+1)*iNum, :]
            test_mean[i,j] = np.mean(test_pi_gi[:,7])

    write_results_txt(val_mean, os.path.join(checkpoint_path, 'mean_val_results.txt'))
    write_results_txt(test_mean, os.path.join(checkpoint_path, 'mean_test_results.txt'))

    name = ['fat', 'heart', 'gut', 'liver', 'lung', 'kidney']
    x_val = [0.55, 0.65, 0.75, 0.85, 0.95]
    x_test = [0.6, 0.7, 0.8, 0.9]
    
    fig = plt.figure(figsize=(5, 4))
    plt.title("Validation Error (MSE)")
    for i in range(pNum):
        plt.plot(x_val, val_mean[i,:], label=name[i])
    plt.legend()
    plt.xlabel('g')
    plt.axis("on")
    plt.savefig(os.path.join(checkpoint_path, 'Fig_Val.png'), bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(5, 4))
    plt.title("Test Error (MSE)")
    for i in range(pNum):
        plt.plot(x_test, test_mean[i,:], label=name[i])
    plt.legend()
    plt.xlabel('g')
    plt.axis("on")
    plt.savefig(os.path.join(checkpoint_path, 'Fig_Test.png'), bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(5, 4))
    plt.title("Estimation Error (MSE)")
    plt.plot(x_val, np.mean(val_mean, axis=0), label='validation')
    plt.plot(x_test, np.mean(test_mean, axis=0), label='test')
    plt.legend()
    plt.xlabel('g')
    plt.axis("on")
    plt.savefig(os.path.join(checkpoint_path, 'Fig_Error.png'), bbox_inches='tight')
    plt.show()


    # show estimation of ua
    # compress the dynamic range of relative errors
    '''
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
    '''
    # show mse of phase function, sorting by g
    val_results = val_results[np.argsort(val_results[:, 6])]    # sorting results according to g 
    test_results = test_results[np.argsort(test_results[:, 6])]

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(2, 1, 1)
    plt.title("Validation")
    plt.plot(val_results[:,7]*10, label='phase_MSE*10')
    plt.plot(val_results[:,6], label='g')
    plt.legend()
    plt.axis("on")
    fig.add_subplot(2, 1, 2)
    plt.title("Test")
    plt.plot(test_results[:,7]*10, label='phase_MSE*10')
    plt.plot(test_results[:,6], label='g')
    plt.xlabel('\n Images (sorted in ascending order by g)')
    plt.legend()
    plt.axis("on")
    plt.savefig(os.path.join(checkpoint_path, 'Fig_phase_g.png'), bbox_inches='tight')
    plt.show()

    # show mse of phase function, sorting by number of non-zeros pixels
    val_results = val_results[np.argsort(val_results[:, 8])]    # sorting results according to num of pixels 
    test_results = test_results[np.argsort(test_results[:, 8])]

    fig = plt.figure(figsize=(10, 8))
    fig.add_subplot(2, 1, 1)
    plt.title("Validation")
    plt.plot(val_results[:,7], label='phase_MSE')
    plt.legend()
    plt.axis("on")
    fig.add_subplot(2, 1, 2)
    plt.title("Test")
    plt.plot(test_results[:,7], label='phase_MSE')
    plt.xlabel('\n Images (sorted in ascending order by number of non-zero pixels)')
    plt.legend()
    plt.axis("on")
    plt.savefig(os.path.join(checkpoint_path, 'Fig_phase_num.png'), bbox_inches='tight')
    plt.show()

    '''
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
    '''
    # --------------------------------------------------------------------------------------------
    fig = show_result_samples(val_data, showFig=True)
    plt.savefig(os.path.join(checkpoint_path, 'Fig_results_samples.png'), bbox_inches='tight')

    # remove the white border around the image, set pad_inches = 0 in the savefig () method.

    print('Done')