# -*- coding:utf-8 -*-
import os

import cv2
import numpy as np
import torch.utils.data as data

kernel = np.ones((3, 3), np.uint8)
infinite = 1e-10
INF = 1e-3


def make_dataset(root):
    imgs = []
    count = 0
    for i in os.listdir(root):
        count += 1
        img = os.path.join(root, i)
        (filename, extension) = os.path.splitext(i)
        if os.path.exists(root.replace("texture", "pure")):
            mask = os.path.join(root.replace("texture", "pure"), filename + '.png')
        imgs.append((img, mask))
    return imgs


class Dataset(data.Dataset):
    def __init__(self, root):
        imgs = make_dataset(root)
        imgs = np.random.permutation(imgs)
        self.imgs = imgs

    def __getitem__(self, index):  # x y m 分别为原图，尺度，pure
        x_path, m_path = self.imgs[index]
        img_x = cv2.imread(x_path)
        img_x = img_x.transpose(2, 0, 1)
        return img_x / 255., img_x / 255.,x_path

    def __len__(self):
        return len(self.imgs)


