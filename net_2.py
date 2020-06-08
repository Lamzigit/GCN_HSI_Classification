# -*- coding:utf-8 -*-
# author: linzhijie time:2020/6/1

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
        seg_y[seg_idx] = label + 1

    for i, _y in enumerate(seg_y):
        if _y == -1:
            mask = (segments == i)
            label = truth[mask].astype(np.int)
            counts = np.bincount(label)
            seg_y[i] = np.argmax(counts)
            if seg_y[i] == 0:
                test_mask[i] = False

    return train_mask, test_mask, seg_y

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

def computeLoss(truth, segement, output):
    shape = segments.shape
    segments = np.reshape(segments, (-1, shape[-1]))
    truth = np.reshape(truth, (-1, shape[-1]))
    new_output = np.zeros((truth.shape[0], output.shape[-1]))

    for idx, c in enumerate(output):
        new_output[segments == idx] = c



class Net(torch.nn.Module):
    def __init__(self, in_channels, cls_num):
        super(Net, self).__init__()
        # self.conv1 = GCNConv(in_channels, 64)
        # self.conv2 = GCNConv(64, 32)
        # self.conv3 = GCNConv(32, cls_num+1)

        self.conv1 = SAGEConv(in_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SAGEConv(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = SAGEConv(32, cls_num + 1)
        self.lc = nn.Linear(32, cls_num)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        # x = self.lc(x)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":

    image = img_as_float(DataReader.PaviauRaw().cube)[:, :, [40, 17, 1]]
    i_sp = image.shape
    image = np.reshape(image, (-1, i_sp[-1]))
    image = StandardScaler().fit_transform(image)
    image = np.reshape(image, i_sp)

    numSegments = 2000

    train_csv = pd.read_csv("data/splitDataset/train/splitPavia_3.csv")
    test_csv = pd.read_csv("data/splitDataset/test/splitPavia_3.csv")
    training_set = train_csv.loc[:, ["row_0", "col_0", "label_0"]].to_numpy()
    testing_set = test_csv.loc[:, ["row_0", "col_0", "label_0"]].to_numpy()

    segments, edge_index = getSuperpixelGraph(image, num_segments=numSegments, compactness=2.5, sigma=2)
    train_mask, test_mask, y = getMaskAndLable(training_set, DataReader.PaviauRaw().truth, segments)
    sp_feature = getSuperpixelFeature(DataReader.PaviauRaw().cube, segments)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(5, 9).to(device)
    x = torch.Tensor(PCA(5).fit_transform(sp_feature))
    edge_index = torch.tensor(edge_index).t().contiguous()

    showSuperpixel(image, segments)

    data = Data(x=x, edge_index=edge_index, test_mask=test_mask, train_mask=train_mask, y=y)
    data.test_mask = torch.tensor(test_mask)
    data.train_mask = torch.tensor(train_mask)
    data.y = torch.tensor(y)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(5000):
        optimizer.zero_grad()
        out = model(data)
        print(out.shape)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        print("loss: ", loss.item())
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
