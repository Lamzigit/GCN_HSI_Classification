# -*- coding:utf-8 -*-
# author: linzhijie time:2020/6/3

# import the necessary packages
from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
from data import DataReader
import numpy as np
import pandas as pd
import os


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
    train_mask = np.full((n_segments, ), False, dtype=bool)
    test_mask = np.full((n_segments, ), True, dtype=bool)
    seg_y = np.full((n_segments, ), -1, dtype=int)

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

if __name__ == "__main__":

    print(os.getcwd())
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    # args = vars(ap.parse_args())
    # # load the image and convert it to a floating point data type
    # image = img_as_float(io.imread(args["image"]))
    # loop over the number of segments
    image = img_as_float(DataReader.KSCRaw().cube)[:, :, [40, 17, 1]]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    i_sp = image.shape
    # apply SLIC and extract (approximately) the supplied number
    # of segments

    # paviau
    image = img_as_float(DataReader.PaviauRaw().cube)[:, :, [100, 65, 3]]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    i_sp = image.shape
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    numSegments = 2600
    segments = slic(image, n_segments=numSegments, compactness=3, sigma=1.5, enforce_connectivity=True, multichannel=True, convert2lab=True)

    train_csv = pd.read_csv("data/splitDataset/train/splitPavia_3.csv")
    test_csv = pd.read_csv("data/splitDataset/test/splitPavia_3.csv")
    training_set = train_csv.loc[:, ["row_0", "col_0", "label_0"]].to_numpy()
    testing_set = test_csv.loc[:, ["row_0", "col_0", "label_0"]].to_numpy()

    train_mask, test_mask, y = getMaskAndLable(training_set, DataReader.PaviauRaw().truth, segments)

    print(np.max(y))
    print(y)


    # ksc
    # image = img_as_float(DataReader.KSCRaw().getCube())[:, :, [40, 17, 1]]
    # image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # i_sp = image.shape
    # # apply SLIC and extract (approximately) the supplied number
    # # of segments
    # numSegments = 4000
    # segments = slic(image, n_segments=numSegments, compactness=3, sigma=1, enforce_connectivity=True, multichannel=True, convert2lab=True)

    # show the output of SLIC

    fig = plt.figure("Superpixels -- %d segments" % (numSegments), figsize=(24, 16))

    ax1 = fig.add_subplot(121)
    ax1.imshow(image, interpolation="none")

    ax = fig.add_subplot(122)
    cube = image
    # cube = (cube - np.min(cube)) / (np.max(cube) - np.min(cube))
    ax.imshow(mark_boundaries(cube, segments), interpolation="none")

    # show the plots
    plt.show()



    # coo = set()
    # dire = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    # for i in range(1, segments.shape[0]):
    #     for j in range(1, segments.shape[1]):
    #         for dx, dy in dire:
    #             if i+dx < segments.shape[0] and i+dx > -1 and \
    #                 j+dy < segments.shape[1] and j+dy > -1 and \
    #                     segments[i, j] != segments[i + dx, j + dy]:
    #                 coo.add((segments[i, j], segments[i + dx, j + dy]))
    #
    # coo = np.asarray(list(coo))
    # print(coo)
    # max = np.max(segments)
    # features = getSuperpixelFeature(image, segments)
    # print(features)
