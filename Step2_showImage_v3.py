import torch
import numpy as np
import pandas as pd
import scipy.io as io

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("Agg")

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

img_path="imageCW_v3"
annotations_file = os.path.join(img_path, "trainDataCW_v3_image.csv")
img_labels = pd.read_csv(annotations_file)

# ==========================================================================================

save_image = False
if save_image:
    # save images, one image per parameter set
    dst_path = "imageCW_v3_image"
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    totalNum = len(img_labels)
    numEachG = 80
    for idx in range(totalNum):
        if idx%numEachG == 0:
            src_file = os.path.join(img_path, img_labels.iloc[idx, 0])
            src_filename = img_labels.iloc[idx, 0]
            dst_filename = src_filename.split(os.sep)[-1].split('.')[0]
            dst_filename = '%s.png' % dst_filename
            dst_file = os.path.join(dst_path, dst_filename)

            ua = img_labels.iloc[idx, 1] 
            us = img_labels.iloc[idx, 2]
            g  = img_labels.iloc[idx, 3]
            pxNum = img_labels.iloc[idx, 4]

            figtitle = 'ua=%.3f, us=%.2f, g=%.2f \n non-zeros=%d' %(ua, us, g, pxNum)
            fig = plt.figure(figsize=(4, 4))
            plt.title(figtitle)
            plt.axis("off")
            img = io.loadmat(src_file).get('rawData')
            img = np.float_power(img, 0.1)
            plt.imshow(img, cmap="hot")
            # plt.show()

            plt.savefig(dst_file)
            plt.close('all')

# ==========================================================================================
# show sample images
# need to uncomment matplotlib.use("Agg")

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
    pxNum = img_labels.iloc[idx, 4]

    img = io.loadmat(img_filename).get('rawData')

    figure1.add_subplot(rows, cols, i+1)
    figtitle = 'ua=%.3f, us=%.2f, g=%.2f \n non-zeros=%d' %(ua, us, g, pxNum)
    plt.title(figtitle)
    plt.axis("off")
    x = img > 0
    plt.imshow(x, cmap="hot")
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()

figure2 = plt.figure(figsize=(8, 4))
for i in range(cols*rows):
    idx = sample_idx[i]
    img_filename = os.path.join(img_path, img_labels.iloc[idx, 0])
    ua = img_labels.iloc[idx, 1] 
    us = img_labels.iloc[idx, 2]
    g  = img_labels.iloc[idx, 3]
    pxNum = img_labels.iloc[idx, 4]

    img = io.loadmat(img_filename).get('rawData')

    figure2.add_subplot(rows, cols, i+1)
    figtitle = 'ua=%.3f, us=%.2f, g=%.2f \n non-zeros=%d' %(ua, us, g, pxNum)
    plt.title(figtitle)
    plt.axis("off")
    # x = np.log(img.squeeze() + 1)
    x = np.float_power(img, 0.1)
    plt.imshow(x, cmap="hot")
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()

figure3 = plt.figure(figsize=(8, 4))
for i in range(cols*rows):
    idx = sample_idx[i]
    img_filename = os.path.join(img_path, img_labels.iloc[idx, 0])
    ua = img_labels.iloc[idx, 1] 
    us = img_labels.iloc[idx, 2]
    g  = img_labels.iloc[idx, 3]
    pxNum = img_labels.iloc[idx, 4]

    img = io.loadmat(img_filename).get('rawData')

    figure3.add_subplot(rows, cols, i+1)
    figtitle = 'ua=%.3f, us=%.2f, g=%.2f \n non-zeros=%d' %(ua, us, g, pxNum)
    plt.title(figtitle)
    plt.axis("off")
    M, N = np.shape(img)
    x = img[int(M/2)][:] + img[int(M/2)+1][:]
    x = np.float_power(x, 0.1)
    plt.plot(x)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()

print('done')