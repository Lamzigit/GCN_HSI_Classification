# -*- coding:utf-8 -*-
# author: linzhijie time:2019/11/5
import numpy as np
from skimage.feature import local_binary_pattern

def getVectorByCoordinate(cube, row, col):

    return np.expand_dims(cube[row, col, :], axis=0)

def getCubeByCoordinate(cube, row, col, radius):

    row_begin = row - radius
    if row_begin < 0:
        row_begin = 0

    row_end = row + radius + 1
    if row_end > cube.shape[0]:
        row_end = cube.shape[0]

    col_begin = col - radius
    if col_begin < 0:
        col_begin = 0

    col_end = col + radius + 1
    if col_end > cube.shape[1]:
        col_end = cube.shape[1]

    cubeOutput = np.zeros((radius*2+1, radius*2+1, cube.shape[-1]))

    try:

        cubeOutput[0:row_end-row_begin, 0:col_end-col_begin, :] = \
            cube[row_begin:row_end, col_begin:col_end, :]
    except ValueError as e:

        a = cube[row_begin:row_end, col_begin:col_end, :]

        print("row: ", row)
        print("row_begin:", row_begin)
        print("row_end:", row_end)

        print("\ncol: ", col)
        print("col_begin:", col_begin)
        print("col_end:", col_end)

        print(cubeOutput[0:row_end-row_begin, 0:col_end-col_begin, :].shape)
        print(cube[row_begin:row_end, col_begin:col_end, :].shape)

        print(e)


    return cubeOutput

# def zeroMeanNormalizeVectorlize(dataCube):
#     band, row, col = dataCube.shape
#
#     dataCube2 = dataCube.contiguous().view(band, row*col)
#
#     std = torch.std(dataCube2, dim=1).view(-1, 1)
#     mean = torch.mean(dataCube2, dim=1).view(-1, 1)
#     dataCube = (dataCube2 - mean) / std
#
#     dataCube = dataCube.view((band, row, col))
#     return dataCube, mean, std

def zeroMeanNormalizeInSpatial(dataCube):
    row, col, band = dataCube.shape

    dataCube2 = dataCube.reshape((row * col, band))

    std = np.expand_dims(np.std(dataCube2, axis=0), axis=0)
    mean = np.expand_dims(np.mean(dataCube2, axis=0), axis=0)
    dataCube = (dataCube2 - mean) / std

    dataCube = dataCube.reshape((row, col, band))
    return dataCube

def zeroMeanNormalizeInSpectral(dataCube):
    row, col, band = dataCube.shape

    dataCube2 = dataCube.reshape((row * col, band))

    std = np.std(dataCube2, axis=1).reshape(-1, 1)
    mean = np.mean(dataCube2, axis=1).reshape(-1, 1)
    dataCube = (dataCube2 - mean) / std

    dataCube = dataCube.reshape((row, col, band))
    return dataCube

def indexToCoordinate(idx, rowNume):
    row = idx % rowNume
    col = idx // rowNume
    return row, col

def kappa_func(matrix):
    n = np.sum(matrix)
    sum_po = 0.0
    sum_pe = 0.0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po * n - sum_pe
    pe = n**2 - sum_pe

    return po/pe

def hsi_Data_Augment(arguments_time, train_pixel_coordinate_set, cube_size):

    arguments_pixel_coordinate_set = np.zeros((3, train_pixel_coordinate_set.shape[1] * arguments_time))
    arguments_pixel_coordinate_count = 0
    direction_set = np.array([[0, 0], [0, -1], [-1, 0], [0, 1], [1, 0], [1, -1], [-1, 1], [1, 1], [-1, -1]])

    for r, c, label in zip(train_pixel_coordinate_set[0, ], train_pixel_coordinate_set[1,], train_pixel_coordinate_set[2]):
        for i in range(arguments_time):

            if r + direction_set[i, 0] < cube_size[0]:
                arguments_pixel_coordinate_set[0, arguments_pixel_coordinate_count] = r + direction_set[i, 0]
            else:
                continue

            if c + direction_set[i, 1] < cube_size[1]:
                arguments_pixel_coordinate_set[1, arguments_pixel_coordinate_count] = c + direction_set[i, 1]
            else:
                continue

            arguments_pixel_coordinate_set[2, arguments_pixel_coordinate_count] = label
            arguments_pixel_coordinate_count += 1

    return arguments_pixel_coordinate_set[:, 0:arguments_pixel_coordinate_count]

def Distance_Data_Augment(arguments_time, train_pixel_coordinate_set, cube, threshold=0.01):
    cube_size = cube.shape
    arguments_set = []
    direction_set = np.array([[0, 0], [0, -2], [-2, 0], [0, 2], [2, 0], [2, -2], [-2, 2], [2, 2], [-2, -2]])

    for r, c, label in train_pixel_coordinate_set:
        center = cube[r, c, :]
        for i in range(arguments_time):
            new_r = r + direction_set[i, 0]
            new_c = c + direction_set[i, 1]
            if new_r < cube_size[0] and new_c < cube_size[1]:
                neibor = cube[new_r, new_c, :]
                distance = np.linalg.norm(center - neibor)
                if distance < threshold:
                    arguments_set.append([new_r, new_c, label])

    arguments_set = np.array(arguments_set)
    return arguments_set

def matlabIndex2Coordinate(train_sample_index, all_data_label, cube_size):

    # ============================================
    # 根据像素的列优先的线性索引，计算像素在直角坐标系的坐标
    # ============================================
    train_pixel_coordinate_set = np.zeros((3, train_sample_index.shape[0]))
    test_pixel_coordinate_set = np.zeros((3, (all_data_label.shape[1] - train_sample_index.shape[0])))
    all_sample_coordinate = np.zeros((3, all_data_label.shape[1]))

    train_i, test_j = 0, 0

    for i, idx_label in enumerate(zip(all_data_label[0], all_data_label[1])):
        idx, label = idx_label
        # 根据像素的列优先的线性索引，计算像素在直角坐标系的坐标
        r, c = indexToCoordinate(idx=idx, rowNume=cube_size[0])
        if idx in train_sample_index:
            # 将 idx 加入 train_pixel_coordinate_set
            train_pixel_coordinate_set[0, train_i] = r
            train_pixel_coordinate_set[1, train_i] = c
            train_pixel_coordinate_set[2, train_i] = label
            train_i += 1
        else:
            # 将 idx 加入 test_pixel_coordinate_set
            test_pixel_coordinate_set[0, test_j] = r
            test_pixel_coordinate_set[1, test_j] = c
            test_pixel_coordinate_set[2, test_j] = label
            test_j += 1
        all_sample_coordinate[0, i] = r
        all_sample_coordinate[1, i] = c
        all_sample_coordinate[2, i] = label

    return train_pixel_coordinate_set, test_pixel_coordinate_set, all_sample_coordinate

def getLocalBinaryPatternCube(cube, p, r, method):

    for i in range(cube.shape[0]):
        cube[i, :, :] = local_binary_pattern(cube[i, :, :], p, r, method)

    return cube



def dataAugmentInClass(train_pixel_coordinate_set, clsNum):

    new_train_pixel_coordinate_set = []

    for cls in range(clsNum):
        cls_sample = train_pixel_coordinate_set[:, train_pixel_coordinate_set[2, :] == cls+1]
        for idx in range(cls_sample.shape[1]):
            sample_1 = cls_sample[:, idx]
            new_train_pixel_coordinate_set.append(np.concatenate((sample_1[:-1], sample_1), axis=0))
            for jdx in range(idx+1, cls_sample.shape[1]):
                sample_2 = cls_sample[:, jdx]
                new_train_pixel_coordinate_set.append(np.concatenate((sample_1[:-1], sample_2), axis=0))


    return np.transpose(np.array(new_train_pixel_coordinate_set))

def dataAugmentInClass2(train_pixel_coordinate_set, clsNum):

    new_train_pixel_coordinate_set = []

    for cls in range(clsNum):
        cls_sample = train_pixel_coordinate_set[:, train_pixel_coordinate_set[2, :] == cls+1]
        for idx in range(cls_sample.shape[1]):
            sample_1 = cls_sample[:, idx]
            new_train_pixel_coordinate_set.append(np.concatenate((sample_1[:-1], sample_1[:-1], sample_1), axis=0))
            for jdx in range(idx+1, cls_sample.shape[1]):
                sample_2 = cls_sample[:, jdx]
                for kdx in range(jdx + 1, cls_sample.shape[1]):
                    sample_3 = cls_sample[:, kdx]
                    new_train_pixel_coordinate_set.append(np.concatenate((sample_1[:-1], sample_2[:-1], sample_3), axis=0))


    return np.transpose(np.array(new_train_pixel_coordinate_set))

def calculateSampleCordinate(groundtruth):
    sample_index = []
    for r in range(groundtruth.shape[0]):
        for c in range(groundtruth.shape[1]):
            if groundtruth[r, c] > -1:
                sample_index.append([r, c, groundtruth[r, c]])
    sample_index = np.array(sample_index)
    return sample_index
