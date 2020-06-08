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
import numpy as np

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

class DataVectorv1(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 cube_radius=4,
                 transform=None,
                 test=False):

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.cube_radius = cube_radius
        self.transform = transform

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]

        self.test = test

    def __len__(self):
        return self.coordinate_label.shape[0]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[item]
        spectral_vector = getVectorByCoordinate(self.cube, int(row), int(col))

        if self.transform != None:
            spectral_vector = self.transform(spectral_vector)
        if self.test:
            return spectral_vector, int(label), row, col
        return spectral_vector, int(label)

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
        return self.coordinate_label.shape[1]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[:, item]
        # row, col = self.indexToCoordinate(row_col, self.cube.shape[0])
        spectral_cube_s = getCubeByCoordinate(self.cube, int(row), int(col), self.cube_radius)
        spectral_cube_s = self.totensor(spectral_cube_s)
        return spectral_cube_s, int(label)

class DataCubev2(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 cube_radius=4,
                 transform=None,
                 test=False):

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.cube_radius = cube_radius
        self.transform = transform

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]
        self.totensor = ToTensor()
        self.test = test


    def __len__(self):
        return self.coordinate_label.shape[0]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[item]
        spectral_cube_s = getCubeByCoordinate(self.cube, int(row), int(col), self.cube_radius)
        spectral_cube_s = self.totensor(spectral_cube_s)
        if self.test:
            return spectral_cube_s, int(label), row, col
        return spectral_cube_s, int(label)

class PCAVectorDataSet(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 cube_radius=1,
                 transform=None,
                 test=False,
                 addData=0,
                 center_w=0.6):

        new_coordinate_label = []
        if not test and addData > 0:

            for coor in coordinate_label:
                new_coordinate_label.append([coor[0], coor[1], coor[2]])
                new_coordinate_label.append([coor[0], coor[1] + 1, coor[2]])
                new_coordinate_label.append([coor[0], coor[1] - 1, coor[2]])
                new_coordinate_label.append([coor[0] + 1, coor[1], coor[2]])
                new_coordinate_label.append([coor[0] - 1, coor[1], coor[2]])

                new_coordinate_label.append([coor[0] + 1, coor[1] + 1, coor[2]])
                new_coordinate_label.append([coor[0] - 1, coor[1] - 1, coor[2]])
                new_coordinate_label.append([coor[0] + 1, coor[1] - 1, coor[2]])
                new_coordinate_label.append([coor[0] - 1, coor[1] + 1, coor[2]])

        if addData > 1:
            for coor in coordinate_label:
                new_coordinate_label.append([coor[0], coor[1] + 2, coor[2]])
                new_coordinate_label.append([coor[0], coor[1] - 2, coor[2]])
                new_coordinate_label.append([coor[0] + 2, coor[1], coor[2]])
                new_coordinate_label.append([coor[0] - 2, coor[1], coor[2]])

                new_coordinate_label.append([coor[0] - 1, coor[1] - 2, coor[2]])
                new_coordinate_label.append([coor[0] - 2, coor[1] - 1, coor[2]])
                new_coordinate_label.append([coor[0] - 2, coor[1] + 1, coor[2]])
                new_coordinate_label.append([coor[0] - 1, coor[1] + 2, coor[2]])

                new_coordinate_label.append([coor[0] + 1, coor[1] + 2, coor[2]])
                new_coordinate_label.append([coor[0] + 2, coor[1] + 1, coor[2]])
                new_coordinate_label.append([coor[0] + 2, coor[1] - 1, coor[2]])
                new_coordinate_label.append([coor[0] + 1, coor[1] - 2, coor[2]])


        if len(new_coordinate_label) > 0:
            coordinate_label = np.array(new_coordinate_label)

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.cube_radius = cube_radius
        self.transform = transform

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]
        self.totensor = ToTensor()
        self.test = test
        wei_len = (cube_radius * 2 + 1) ** 2
        wei_uni = (1 - center_w) / (wei_len - 1)
        wei = [wei_uni for i in range(wei_len)]
        wei[wei_len // 2 + 1] = center_w
        self.neigh_weight = torch.as_tensor(wei, dtype=torch.double).unsqueeze(0)


    def __len__(self):
        return self.coordinate_label.shape[0]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[item]
        spectral_cube_s = getCubeByCoordinate(self.cube, int(row), int(col), self.cube_radius)
        spectral_cube_s = torch.as_tensor(spectral_cube_s, dtype=torch.double).reshape(-1, spectral_cube_s.shape[-1])
        spectral_cube_s = torch.mm(self.neigh_weight, spectral_cube_s)
        if self.test:
            return spectral_cube_s, int(label), row, col
        return spectral_cube_s, int(label)

class DataCubev3(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 cube_radius=4,
                 groundtruth=None,
                 transform=None,
                 test=False):

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.cube_radius = cube_radius
        self.transform = transform
        self.gt = groundtruth

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]
        self.test = test
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
        if self.test:
            return spectral_vector, spectral_cube_b, spectral_cube_s, int(label), row, col
        return spectral_vector, spectral_cube_b, spectral_cube_s, int(label)

class MLPDataset(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 cube_radius=4,
                 transform=None):

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.cube_radius = cube_radius
        self.transform = transform

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]

    def __len__(self):
        return self.coordinate_label.shape[1]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[:, item]
        spectral_vector = getVectorByCoordinate(self.cube, int(row), int(col))

        if self.transform != None:
            spectral_vector = self.transform(spectral_vector)

        return spectral_vector, int(label)

class SaSeLSTMDataset(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 pca_cbue,
                 cube_radius=4,
                 transform=None,
                 test=False):

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.pca_cube = pca_cbue
        self.cube_radius = cube_radius
        self.transform = transform

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]
        self.test = test


    def __len__(self):
        return self.coordinate_label.shape[0]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[item]
        spectral_vector = getVectorByCoordinate(self.cube, int(row), int(col))
        pca_cube = getCubeByCoordinate(self.pca_cube, int(row), int(col), radius=30)

        if self.transform != None:
            spectral_vector = self.transform(spectral_vector)
        if self.test:
            return spectral_vector, pca_cube, int(label), row, col
        return spectral_vector, pca_cube, int(label)


class LeNetDataSet(Dataset):

    def __init__(self,
                 coordinate_label,
                 cube,
                 cube_radius=15,
                 transform=None,
                 test=False):

        self.coordinate_label = coordinate_label
        self.cube = cube
        self.cube_radius = cube_radius
        self.transform = transform

        self.col_num = cube.shape[2]
        self.row_num = cube.shape[1]
        self.test = test

    def __len__(self):
        return self.coordinate_label.shape[0]

    def __getitem__(self, item):
        row, col, label = self.coordinate_label[item]
        spectral_cube = np.transpose(getCubeByCoordinate(self.cube, int(row), int(col), radius=16), (2, 0, 1))

        if self.transform != None:
            spectral_cube = self.transform(spectral_cube)
        if self.test:
            return spectral_cube, int(label), row, col
        return spectral_cube, int(label)

class SuSsseSet(Dataset):

    def __init__(self,
                 SuSsseCube,
                 label_ers,
                 meanPosition):
        self.SuSsseCube = SuSsseCube
        self.label_ers = label_ers
        self.meanPosition = meanPosition
        self.coo_x, self.coo_y = np.meshgrid(np.arange(label_ers.shape[0]), np.arange(label_ers.shape[1]))

    def __len__(self):
        return self.SuSsseCube.shape[0]

    def __getitem__(self, item):
        SuPixel = self.SuSsseCube[item]
        # SuPosition = self.meanPosition[item]
        coo_x = self.coo_x[self.label_ers == item]
        coo_y = self.coo_y[self.label_ers == item]

        return SuPixel, coo_x, coo_y