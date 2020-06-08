# -*- coding:utf-8 -*-
# author: linzhijie time:2019/11/9
import scipy.io as sio
import os.path as osp
import pandas as pd
import numpy as np

from data.utils import hsi_Data_Augment, Distance_Data_Augment, zeroMeanNormalizeInSpatial
if __name__ == "__main__":
    flag = 1
    root_dir = "splitDataset/"
    cube_file = ""
    groundtruth_file = ""
    key = ""
    split_file = ""
    cube_key = ""
    max_sample_num = 3
    # threshold = 20 * 0.30
    threshold = 1.3 * 0.60
    if flag == 1:
        groundtruth_file = "../data/label/PaviaU_gt.mat"
        cube_file = "../data/original_cube/PaviaU.mat"
        key = "paviaU_gt"
        cube_key = "paviaU"
        split_file = "splitPavia_"
    elif flag == 2:
        groundtruth_file = "../data/label/Salinas_gt.mat"
        cube_file = "../data/original_cube/Salinas.mat"
        key = "salinas_gt"
        cube_key = "salinas"
        split_file = "splitSalinas_"
    elif flag == 3:
        groundtruth_file = "../data/label/Indian_pines_gt.mat"
        cube_file = "../data/original_cube/Indian_pines.mat"
        key = "indian_pines_gt"
        cube_key = "indian_pines"
        split_file = "splitIndian_"
        max_sample_num = 3
    elif flag == 4:
        groundtruth_file = "../data/label/KSC_gt.mat"
        cube_file = "../data/original_cube/KSC.mat"
        key = "KSC_gt"
        cube_key = "KSC"
        split_file = "splitKSC_"

    cube = zeroMeanNormalizeInSpatial(sio.loadmat(cube_file)[cube_key].astype(int))
    groundtruth = sio.loadmat(groundtruth_file)[key].astype(int)
    groundtruth -= 1
    for sample_num in range(3, max_sample_num+1):

        train_save_name = split_file + str(sample_num) + ".csv"
        train_save_name = osp.join(root_dir, "train", train_save_name)
        train_csv = pd.read_csv(train_save_name)
        correct = []
        for iter in range(10):
            training_set = train_csv.loc[:, ["row_"+str(iter), "col_"+str(iter), "label_"+str(iter)]].to_numpy()
            training_set = Distance_Data_Augment(9, training_set, cube, threshold=threshold)
            count = 0.0
            for idx in range(len(training_set)):
                r, c, l = training_set[idx]
                if groundtruth[r, c] == l:
                    count += 1
            print("train num: %2d, argument: %4d, iter: %2d, correct: %6f" %
                  (sample_num, len(training_set), iter, count/len(training_set)))
            correct.append(count/len(training_set))
        correct = np.array(correct)
        mean = np.mean(correct)
        print("##########################")
        print("average corrected rate on %2d sample: %6f" % (sample_num, mean))
        print("##########################")
