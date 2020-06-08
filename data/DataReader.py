# -*- coding:utf-8 -*-
# author: linzhijie time:2019/7/21

import torch
import numpy as np
import scipy.io as sio
import os
# if os.getcwd().find("/data"):
#     print("changing dir")
#     os.chdir("..")

class DataReader(object):

    def __init__(self):
        self.data_cube = None
        self.g_truth = None

    @property
    def cube(self):
        return self.data_cube

    @property
    def truth(self):
        return self.g_truth


class PaviauRaw(DataReader):

    def __init__(self):
        super(PaviauRaw, self).__init__()

        raw_data_package = sio.loadmat("data/Pavia.mat")
        self.data_cube = raw_data_package["paviaU"].astype(float)
        truth = sio.loadmat("data/label/PaviaU_gt.mat")
        self.g_truth = truth["paviaU_gt"].astype(float)


class IndianRaw(DataReader):

    def __init__(self):
        super(IndianRaw, self).__init__()

        # 加载原始数据包
        raw_data_package = sio.loadmat("data/Indian_pines.mat")
        # raw_data_package = sio.loadmat("/home/huawei/linzhijie/2019_Dilation-CNN/data/original_cube/Indiana185_Ref.mat")

        # 读取原始数据立方体
        # self.data_cube = np.reshape(raw_data_package["x"].astype(float), (145, 145, 185)).transpose((1, 0, 2))
        self.data_cube = raw_data_package["indian_pines"].astype(float)

        # Indian的mat格式数据的形状是[波段数，样本数]，输入网络前需要将其转化为立方体
        # 在matlab中reshape是以列优先，在python中是行优先，
        # 因此需要将数据转化为[band_num, col, row], 然后转置为[band_num, row, col]
        # raw_data_cube = raw_data_cube.view(band_num, col_num, row_num).transpose_(1, 2)
        # self.raw_data_cube = np.transpose(self.raw_data_cube, (2, 0, 1))


class KSCRaw(DataReader):
    def __init__(self):
        super(KSCRaw, self).__init__()

        raw_data_package = sio.loadmat("data/KSC.mat")
        self.data_cube = raw_data_package["KSC"].astype(float)
        truth = sio.loadmat("data/label/KSC_gt.mat")
        self.g_truth = truth["KSC_gt"].astype(float)


class SalinasRaw(DataReader):
    def __init__(self):
        super(SalinasRaw, self).__init__()

        # 加载原始数据包
        raw_data_package = sio.loadmat("data/Salinas.mat")

        # 读取原始数据立方体
        self.data_cube = raw_data_package["salinas"].astype(float)


class HoustonRaw(DataReader):
    def __init__(self):
        super(HoustonRaw, self).__init__()

        # 加载原始数据包
        raw_data_package = sio.loadmat("data/Houston.mat")

        # 读取原始数据立方体
        self.data_cube = raw_data_package["houston"].astype(float)


if __name__ == "__main__":
    print(os.getcwd())

    reader = KSCRaw()
    a = reader.cube
    print(a.shape)
    a = reader.truth
    print(a.shape)
