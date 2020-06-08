# -*- coding:utf-8 -*-
# author: linzhijie time:2020/6/8

from torch_geometric.nn import GCNConv

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import numpy as np
from data import DataReader
from skimage.util import img_as_float
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def getSuperpixelGraph(image, num_segments, compactness=300, sigma=3.):
    segments = slic(image, n_segments=num_segments, compactness=compactness, sigma=sigma,
                    multichannel=True, convert2lab=True)
    # show the output of SLIC
    coo = set()
    dire = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    for i in range(1, segments.shape[0]):
        for j in range(1, segments.shape[1]):
            for dx, dy in dire:
                if -1 < i + dx < segments.shape[0] and \
                        -1 < j + dy < segments.shape[1] and \
                        segments[i, j] != segments[i + dx, j + dy]:
                    coo.add((segments[i, j], segments[i + dx, j + dy]))

    coo = np.asarray(list(coo))
    return segments, coo

def getSuperpixelFeature(image, segments, mode="mean"):
    features = []
    segments = segments.flatten()
    image = np.reshape(image, (-1, image.shape[-1]))
    method = np.mean
    if mode == "max":
        method = np.max
    if mode == "min":
        method = np.min

    for i in range(int(np.max(segments)) + 1):
        features.append(method(image[segments == i], axis=0))
    return np.asarray(features)

def getMaskAndLable(train_idx, truth, segments):
    """

    Parameters
    ----------
    train_idx : [sample, coo_lable]
    truth : shape [w, h]
    segments : shape [w, h]

    Returns
    -------

    """
    n_segments = np.max(segments) + 1
    train_mask = np.full((n_segments,), False, dtype=bool)
    test_mask = np.full((n_segments,), True, dtype=bool)
    seg_y = np.full((n_segments,), -1, dtype=int)

    for r, c, label in train_idx:
        seg_idx = segments[r, c]
        train_mask[seg_idx] = True
        test_mask[seg_idx] = False
        seg_y[seg_idx] = label

    for i, _y in enumerate(seg_y):
        if _y == -1:
            mask = (segments == i)
            label = truth[mask].astype(np.int)
            counts = np.bincount(label)
            seg_y[i] = 0 if len(counts)==1 else np.argmax(counts[1:])
            if seg_y[i] == 0:
                test_mask[i] = False

    seg_y[i] -= 1
    return train_mask, test_mask, seg_y

def getGlobalMask(truth, test_idx):
    mask = np.zeros_like(truth, dtype=bool)
    for r, c, _ in test_idx:
        mask[r, c] = True
    return mask.flatten()

def showSuperpixel(image, segments):
    numSegments = np.max(segments) + 1
    fig = plt.figure("Superpixels -- %d segments" % (numSegments), figsize=(24, 16))

    ax1 = fig.add_subplot(121)
    ax1.imshow(image, interpolation="none")

    ax = fig.add_subplot(122)
    cube = image
    # cube = (cube - np.min(cube)) / (np.max(cube) - np.min(cube))
    ax.imshow(mark_boundaries(cube, segments), interpolation="none")

    # show the plots
    plt.show()

def computeLoss(truth, segments, output, global_test_mask=None):
    new_output = torch.zeros((truth.shape[0], output.shape[-1]), dtype=torch.float).to(segments.device)
    for idx, c in enumerate(output):
        new_output[segments == idx] = c
    if not global_test_mask is None:
        truth = truth[global_test_mask] - 1
        new_output = new_output[global_test_mask]

    return F.nll_loss(new_output, truth)
