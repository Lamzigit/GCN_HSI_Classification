# -*- coding:utf-8 -*-
# author: linzhijie time:2020/6/1

from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import numpy as np

dataset = Planetoid(root='/tmp/Cora', name='Cora')


def getSuperpixelGraph(image, num_segments, compactness=300, sigma=3):
    segments = slic(image, n_segments=num_segments, compactness=300, sigma=3, multichannel=True)
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


def getSuperpixelFeature(image, segments, mode="max"):
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




class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

if __name__ == "__main__":



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        print("loss: ", loss.item())
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))
