
# -*- coding:utf-8 -*-
# author: linzhijie time:2019/11/8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import os
import os.path as osp
import time
from data.utils import zeroMeanNormalizeInSpatial, kappa_func, \
    matlabIndex2Coordinate, getVectorByCoordinate, dataAugmentInClass, dataAugmentInClass2, hsi_Data_Augment
import data.DataReader as DataReader
import pandas as pd


if __name__ == "__main__":

    flag = 1
    root_dir = "splitDataset/"
    groundtruth_file = None
    key = None
    split_file = None
    max_sample_num = 20

    if flag == 1:
        groundtruth_file = "../data/label/PaviaU_gt.mat"
        key = "paviaU_gt"
        split_file = "splitPavia_"
    elif flag == 2:
        groundtruth_file = "../data/label/Salinas_gt.mat"
        key = "salinas_gt"
        split_file = "splitSalinas_"
    elif flag == 3:
        groundtruth_file = "../data/label/Indian_pines_gt.mat"
        key = "indian_pines_gt"
        split_file = "splitIndian_"
        max_sample_num = 15
    elif flag == 4:
        groundtruth_file = "../data/label/KSC_gt.mat"
        key = "KSC_gt"
        split_file = "splitKSC_"

    np.random.seed(0)

    groundtruth = sio.loadmat(groundtruth_file)[key].astype(int)
    groundtruth -= 1
    sample_index = []
    for r in range(groundtruth.shape[0]):
        for c in range(groundtruth.shape[1]):
            if groundtruth[r, c] > -1:
                sample_index.append([r, c, groundtruth[r, c]])
    sample_index = np.array(sample_index)
    label_num = len(np.unique(sample_index[:, 2]))

    # for sample_num in range(3, max_sample_num+1):
    for sample_num in [50]:

        header = []
        for iter in range(10):
            header.extend(["row_"+str(iter), "col_"+str(iter), "label_"+str(iter)])

        train_dataset = pd.DataFrame(columns=header)
        test_dataset = pd.DataFrame(columns=header)
        train_save_name = split_file + str(sample_num) + ".csv"
        train_save_name = osp.join(root_dir, "train", train_save_name)

        test_save_name = split_file + str(sample_num) + ".csv"
        test_save_name = osp.join(root_dir, "test", test_save_name)

        for iter in range(10):

            training_set = []
            testing_set = []
            for cls in range(label_num):
                samples = sample_index[sample_index[:, 2] == cls]
                # train_num = int(0.9 * samples.shape[0])
                # test_num = samples.shape[1] - train_num
                index = np.arange(samples.shape[0])
                np.random.shuffle(index)
                training_set.append(samples[index[:sample_num]])
                testing_set.append(samples[index[sample_num:]])
            training_set = np.concatenate(training_set, axis=0)
            testing_set = np.concatenate(testing_set, axis=0)
            for i, colkey in enumerate(["row_"+str(iter), "col_"+str(iter), "label_"+str(iter)]):
                train_dataset.loc[:, colkey] = training_set[:, i]
                test_dataset.loc[:, colkey] = testing_set[:, i]
        train_dataset.to_csv(train_save_name)
        test_dataset.to_csv(test_save_name)




