import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import cv2
import numpy as np
from skimage.transform import resize


class Dataset(Dataset):
    def __init__(self, data_dir, transform=None, data_type='both'):
        """
        Custom Dataset Class
        :param data_dir: data directory paths [type: str]
        :param transform: transforms to be applied to the images [type: boolean]
        :param data_type: data type ['both', 'a', 'b']
        """
        self.data_dir_A = os.path.join(data_dir, 'A')
        self.data_dir_B = os.path.join(data_dir, 'B')
        self.transform = transform
        self.data_type = data_type

        if os.path.exists(self.data_dir_A):
            data_A_list = os.listdir(self.data_dir_A)
            data_A_list = [f for f in data_A_list if f.endswith('jpg') | f.endswith('png') | f.endswith('jpeg')]
            data_A_list.sort()
        else:
            list_A_list = list()

        if os.path.exists(self.data_dir_B):
            data_B_list = os.listdir(self.data_dir_B)
            data_B_list = [f for f in data_B_list if f.endswith('jpg') | f.endswith('png') | f.endswith('jpeg')]
            data_B_list.sort()

        else:
            data_B_list = list()

        self.data_A_list = data_A_list
        self.data_B_list = data_B_list

    def __len__(self):
        if self.data_type == 'both':
            if len(self.data_A_list) < len(self.data_B_list):
                return len(self.data_A_list)
            else:
                return len(self.data_B_list)
        elif self.data_type == 'a':
            return len(self.data_A_list)
        elif self.data_type == 'b':
            return len(self.data_B_list)

    def __getitem__(self, idx):
        data = dict()
        if self.data_type == 'a' or self.data_type == 'both':
            img_A = cv2.imread(os.path.join(self.data_dir_A, self.data_A_list[idx]))
            img_A = cv2.COLOR_BGR2RGB(img_A, cv2.COLOR_BGR2RGB)

            if img_A.ndim == 2: # channel이 없을 경우.
                img_A = img_A[:, :, np.newaxis]
            if img_A.dtype == np.uint8:
                img_A /= 255.0

            data['img_A'] = img_A

        if self.data_type == 'b' or self.data_type == 'both':
            img_B = cv2.imread(os.path.join(self.data_dir_B, self.data_B_list[idx]))
            img_B = cv2.COLOR_BGR2RGB(img_B, cv2.COLOR_BGR2RGB)

            if img_B.ndim == 2:
                img_B = img_B[:, :, np.newaxis]
            if img_B.dtype == np.uint8:
                img_B /= 255.0

            data['img_B'] = img_B

        # apply transforms
        if self.transform:
            data = self.transform(data)

        # numpy converted to tensor
        data = ToTensor(data)

        return data

# ============================================ #
#              Transforms 구현                  #
# ============================================ #

class ToTensor(object):
    def __call__(self, data):
        """
        :param data: type is dict ['img_A': ____, 'img_B': ____]
        :return: numpy of directory converted to tensor (data)
        """
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype('float32') # H, W, C -> C, H, W
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        :param data: type is dict ['img_A': ____, 'img_B': ____]
        :return: numpy of directory converted to tensor (data)
        """
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data

class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        """
        :param data: type is dict ['img_A': ____, 'img_B': ____]
        :return: numpy of directory converted to tensor (data)
        """
        for key, value in data.items():
            data[key] = resize(image=value, output_shape=(self.shape[0], self.shape[1], self.shape[2]))

        return data

class RandomFlip(object):
    def __call__(self, data):
        """
        :param data: type is dict ['img_A': ____, 'img_B': ____]
        :return: numpy of directory converted to tensor (data)
        """
        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)
        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)
        return data

class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        """
        :param data: type is dict ['img_A': ____, 'img_B': ____]
        :return: numpy of directory converted to tensor (data)
        """
        keys = list(data.keys())

        h, w = data[keys[0]].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data












