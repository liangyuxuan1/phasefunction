import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import scipy.io as io


# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

img_path="imageCW"
annotations_file = os.path.join(img_path, "trainDataCW.csv")
img_labels = pd.read_csv(annotations_file)

figure1 = plt.figure(figsize=(8, 8))
cols, rows = 3, 5
for i in range(1, cols * rows):
    sample_idx = 1 + (i - 1)*500
    img_filename = os.path.join(img_path, img_labels.iloc[sample_idx, 0])
    ua = img_labels.iloc[sample_idx, 1]
    us = img_labels.iloc[sample_idx, 2]
    g = img_labels.iloc[sample_idx, 3]

    img = io.loadmat(img_filename).get('rawData')

    figure1.add_subplot(rows, cols, i)
    figtitle = 'ua=%.4f,us=%.4f,g=%.4f' %(ua, us, g)
    plt.title(figtitle)
    plt.axis("off")
    x = (img.squeeze() > 0)*128.0
    plt.imshow((x), cmap="hot")
    #np.log(np.log(img.squeeze()+1)+1
plt.show()


figure2 = plt.figure(figsize=(8, 8))
for i in range(1, cols * rows):
    sample_idx = 1 + (i - 1)*500
    img_filename = os.path.join(img_path, img_labels.iloc[sample_idx, 0])
    ua = img_labels.iloc[sample_idx, 1]
    us = img_labels.iloc[sample_idx, 2]
    g = img_labels.iloc[sample_idx, 3]

    img = io.loadmat(img_filename).get('rawData')

    figure2.add_subplot(rows, cols, i)
    figtitle = 'ua=%.4f,us=%.4f,g=%.4f' %(ua, us, g)
    plt.title(figtitle)
    plt.axis("off")
    plt.imshow(np.log(np.log(img.squeeze()+1)+1), cmap="hot")
plt.show()