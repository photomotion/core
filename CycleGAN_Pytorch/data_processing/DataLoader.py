import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import cv2
import numpy as np


class Dataset(Dataset):
    def __init__(self, data_dir, transform=None, data_type='both'):
        """
        Custom Dataset Class
        :param data_dir: data directory paths
        :param transform: transforms to be applied to the images
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



