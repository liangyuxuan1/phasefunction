import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import scipy.io as io


# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

img_path="imageCW_v3"
annotations_file = os.path.join(img_path, "trainDataCW_v3_image.csv")
img_labels = pd.read_csv(annotations_file)

cols, rows = 4, 2
numEachUaGroup = 9*21*80
sample_idx = np.zeros(cols*rows, dtype=np.int32)
for i in range (cols*rows):
    sample_idx[i] = np.random.randint(numEachUaGroup) + i*numEachUaGroup

figure1 = plt.figure(figsize=(8, 4))
for i in range(cols*rows):
    idx = sample_idx[i]
    img_filename = os.path.join(img_path, img_labels.iloc[idx, 0])
    ua = img_labels.iloc[idx, 1] 
    us = img_labels.iloc[idx, 2]
    g  = img_labels.iloc[idx, 3]

    img = io.loadmat(img_filename).get('rawData')

    figure1.add_subplot(rows, cols, i+1)
    figtitle = 'ua=%.3f, us=%.2f, g=%.1f' %(ua, us, g)
    plt.title(figtitle)
    plt.axis("off")
    x = img.squeeze() > 0
    plt.imshow((x), cmap="hot")
    #np.log(np.log(img.squeeze()+1)+1
plt.show()

figure2 = plt.figure(figsize=(8, 4))
for i in range(cols*rows):
    idx = sample_idx[i]
    img_filename = os.path.join(img_path, img_labels.iloc[idx, 0])
    ua = img_labels.iloc[idx, 1] 
    us = img_labels.iloc[idx, 2]
    g  = img_labels.iloc[idx, 3]

    img = io.loadmat(img_filename).get('rawData')

    figure2.add_subplot(rows, cols, i+1)
    figtitle = 'ua=%.3f, us=%.2f, g=%.1f' %(ua, us, g)
    plt.title(figtitle)
    plt.axis("off")
    # x = np.log(img.squeeze() + 1)
    x = np.float_power(img.squeeze(), 0.1)
    plt.imshow((x), cmap="hot")
plt.show()
