# -*- coding:utf-8 -*-
# author: linzhijie time:2019/11/5
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import CenterCrop, ToTensor
from data.utils import kappa_func, \
    matlabIndex2Coordinate, getVectorByCoordinate, hsi_Data_Augment, getCubeByCoordinate

class DataIn(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 cube_radius=4,
                 groundtruth=None,
                 transform=None):

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.cube_radius = cube_radius
        self.transform = transform
        self.gt = groundtruth

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]
        if coordinate_label.shape[0] > 3:
            self.training = True
        else:
            self.training = False
        self.totensor = ToTensor()

    def indexToCoordinate(self, idx, rowNume):
        row = idx % rowNume
        col = idx // rowNume
        return row, col

    def __len__(self):
        return self.coordinate_label.shape[0]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[item]
        # row, col = self.indexToCoordinate(row_col, self.cube.shape[0])
        # gt = self.gt[row, col]
        # if int(gt) == int(label):
        #     print("yes")
        # else:
        #     print("no ", "gt: ", gt, " label: ", label)
        spectral_vector = getVectorByCoordinate(self.cube, int(row), int(col))
        spectral_cube_b = getCubeByCoordinate(self.cube, row, col, self.cube_radius)
        spectral_cube_s = getCubeByCoordinate(self.cube, row, col, self.cube_radius-2)

        if self.transform != None:
            spectral_vector = self.transform(spectral_vector)
        spectral_cube_b = self.totensor(spectral_cube_b)
        spectral_cube_s = self.totensor(spectral_cube_s)

        return spectral_vector, spectral_cube_b, spectral_cube_s, int(label)


class DataCube(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 cube_radius=4,
                 groundtruth=None,
                 transform=None):

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.cube_radius = cube_radius
        self.transform = transform
        self.gt = groundtruth

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]
        if coordinate_label.shape[0] > 3:
            self.training = True
        else:
            self.training = False
        self.totensor = ToTensor()

    def indexToCoordinate(self, idx, rowNume):
        row = idx % rowNume
        col = idx // rowNume
        return row, col

    def __len__(self):
        return self.coordinate_label.shape[0]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[item]
        spectral_cube_s = getCubeByCoordinate(self.cube, row, col, self.cube_radius-2)
        spectral_cube_s = self.totensor(spectral_cube_s)
        return spectral_cube_s, int(label)

class DataCubev2(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 cube_radius=4,
                 groundtruth=None,
                 transform=None):

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.cube_radius = cube_radius
        self.transform = transform
        self.gt = groundtruth

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]
        if coordinate_label.shape[0] > 3:
            self.training = True
        else:
            self.training = False
        self.totensor = ToTensor()

    def indexToCoordinate(self, idx, rowNume):
        row = idx % rowNume
        col = idx // rowNume
        return row, col

    def __len__(self):
        return self.coordinate_label.shape[0]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[item]
        # row, col = self.indexToCoordinate(row_col, self.cube.shape[0])
        # gt = self.gt[row, col]
        # if int(gt) == int(label):
        #     print("yes")
        # else:
        #     print("no ", "gt: ", gt, " label: ", label)
        spectral_vector = getVectorByCoordinate(self.cube, int(row), int(col))
        spectral_cube_b = getCubeByCoordinate(self.cube, row, col, self.cube_radius)
        spectral_cube_s = getCubeByCoordinate(self.cube, row, col, self.cube_radius-2)

        if self.transform != None:
            spectral_vector = self.transform(spectral_vector)
        spectral_cube_b = self.totensor(spectral_cube_b)
        spectral_cube_s = self.totensor(spectral_cube_s)

        return spectral_vector, spectral_cube_b, spectral_cube_s, int(label)