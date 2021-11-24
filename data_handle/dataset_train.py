
# -*- coding:utf-8 -*-
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data

max_distance = 18
distance_max = 18
angle_num = 6
kernel = np.ones((3, 3), np.uint8)
infinite = 1e-10
INF = 1e-3
IMG_H = 224
angle_distance = np.ones((IMG_H, IMG_H, angle_num))
IS_TRAIN_FEN = False
IS_TRAIN_CEN = False
IS_TRAIN_ISN = True
IS_FINETUNED = False

def get_angle_distance_detail():
    angle = [360*i/angle_num + 180/angle_num for i in range(angle_num)]
    rad = np.array([math.radians(i) for i in angle])
    for i in range(IMG_H):  # i is height j is weight
        for j in range(IMG_H):
            for k in range(angle_num):
                if np.sin(rad[k]) > 0:
                    max_h = IMG_H - i - 1
                    max_dis_h = max_h / np.sin(rad[k])
                else:
                    max_h = i
                    max_dis_h = max_h / np.sin(rad[k])
                if abs(np.cos(rad[k])) < INF:
                    max_dis_w = 999
                elif np.cos(rad[k]) > 0:
                    max_w = IMG_H - j - 1
                    max_dis_w = max_w / np.cos(rad[k])
                else:
                    max_w = j
                    max_dis_w = max_w / np.cos(rad[k])
                angle_distance[i, j, k] = (min(abs(max_dis_h), abs(max_dis_w)) + 1) / max_distance


get_angle_distance_detail()


def get_filter_detail(f):
    f = f.copy()
    angle = np.array([i * 360 / angle_num for i in np.arange(0, angle_num)])
    angle = torch.from_numpy(angle)
    angle = angle.int()
    rad = np.array([math.radians(i) for i in angle])
    rad = torch.from_numpy(rad)  # 角度转弧度
    angle = np.arange(360)
    rad = np.array([math.radians(i) for i in angle])
    mx = np.cos(rad) * distance_max
    mx = np.ceil(mx)
    mx = np.array(mx, np.int)
    # mx = mx.int()
    my = np.sin(rad) * distance_max
    my = np.ceil(my)
    my = np.array(my, np.int)
    for j in range(angle_num):
        t = np.zeros((2 * distance_max + 1, 2 * distance_max + 1, 3), np.float)
        for k in range(np.int(j * 360 / angle_num) + 1, np.int((j + 1) * 360 / angle_num)):
            cv2.line(t, (distance_max, distance_max), (distance_max + mx[k], distance_max + my[k]), (255, 255, 255),
                     1,
                     cv2.LINE_AA)  # cv2.LINE_AA
        t = cv2.dilate(t, kernel, iterations=1)
        t = cv2.erode(t, kernel, iterations=1)
        t = t[:, :, 0] / sum(sum(t[:, :, 0]))
        f[:, :, j] = t
        tt = f[:, :, j]
    return f


f = np.zeros((2 * distance_max + 1, 2 * distance_max + 1, angle_num), np.float)
f = get_filter_detail(f)


def check_edge(img, dis):  # img是pure
    mask = (dis > angle_distance).astype(np.float)
    dis = np.where(mask, 1.0, dis)
    return dis


def classify(img):
    h, w, a = img.shape
    gt = np.zeros((3, a, h, w))  # 第0维为边缘 第1维为边缘附近 第2维为不靠近边缘
    T = np.ones((3, a, h, w))
    Z = np.zeros((h, w))
    for i in range(a):
        gt[0, i, :, :] = np.where((img[:, :, i] < 2 / distance_max), T[0, i, :, :], gt[0, i, :, :])
        gt[1, i, :, :] = np.where(((img[:, :, i] >= 2 / distance_max) * (img[:, :, i] <= 0.9)), T[1, i, :, :],
                                  gt[1, i, :, :])
        gt[2, i, :, :] = np.where((img[:, :, i] > 0.9), T[2, i, :, :], gt[2, i, :, :])
    gt = gt.max(1)
    t1 = (gt[0, :, :] >= gt[1, :, :])
    t2 = ((gt[0, :, :] + gt[1, :, :]) >= gt[2, :, :])
    gt[1, :, :] = np.where(t1, Z, gt[1, :, :])
    gt[2, :, :] = np.where(t2, Z, gt[2, :, :])
    return gt


def make_dataset(root):
    imgs = []
    count = 0
    for i in os.listdir(root):
        count += 1
        img = os.path.join(root, i)
        (filename, extension) = os.path.splitext(i)
        if os.path.exists(root.replace("texture", "pure")):
            mask = os.path.join(root.replace("texture", "pure"), filename + '.png')
        if os.path.exists(root.replace("texture", "twin")):
            edge = os.path.join(root.replace("texture", "twin"), filename + '.png')
        if os.path.exists(root.replace("texture", "result8-16")):
            if os.path.exists(os.path.join(root.replace("texture", "result8-16") + '/' + i.split('.')[0] + '.npy')):
                distance_gt = os.path.join(root.replace("texture", "result8-16") + '/' + i.split('.')[0] + '.npy')
            else:
                distance_gt = ''
        if os.path.exists(root.replace("texture", "twin")):
            imgs.append((img, distance_gt, mask, edge))
        else:
            imgs.append((img, distance_gt, mask))
    return imgs


def get_bigger(img):
    img = img.transpose(1, 2, 0)
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    img = img.transpose(2, 0, 1)
    return img


def make_band_gt(edge):
    k_size = max_distance*2+1
    edge1 = cv2.GaussianBlur(edge, (k_size, k_size), 0, 0, cv2.BORDER_DEFAULT)  # 此时这个边界已经是纹理了
    band = (edge1 != 0).astype(np.float)
    edge = cv2.GaussianBlur(band, (max_distance*2+1, max_distance*2+1), 0, 0, cv2.BORDER_DEFAULT)
    b = (band == 0).astype(np.float)
    edge = np.where(b, b, edge)
    edge = edge * edge
    return band, edge


class Dataset(data.Dataset):
    def __init__(self, root, train_transform=None, test=False, valid=False, mean=False, stage=1):
        imgs = make_dataset(root)
        imgs_num = len(imgs)
        imgs = np.random.permutation(imgs)
        self.stage = stage
        if valid:
            self.imgs = imgs[:int(0.1 * imgs_num)]
        elif not test:
            self.imgs = imgs[int(0.3 * imgs_num):]
        elif test:
            self.imgs = imgs
        self.train_transform = train_transform
        self.test = test
        self.mean = mean

    def __getitem__(self, index):  # x y m 分别为原图，尺度，pure
        if len(self.imgs[0]) == 3:
            x_path, y_path, m_path = self.imgs[index]
        else:
            x_path, y_path, m_path, edge = self.imgs[index]
        try:
            img_x = cv2.imread(x_path)
            img_x = img_x.transpose(2, 0, 1)
        except:
            a = 1
        if not self.test:
            img_z = cv2.imread(m_path)
            edge = cv2.imread(edge)
            img_z = img_z.transpose(2, 0, 1)
        else:
            img_z = None
        if self.mean:
            img_z = cv2.imread(m_path)
            img_z = img_z.transpose(2, 0, 1)
        if not self.test:
            img_y = np.load(y_path)
            img_y = img_y / max_distance  # 归为0~1之间  是否需要先减1？

            size = 224
            _, x, y = img_x.shape
            x_rand = random.randint(0, x - size)
            y_rand = random.randint(0, y - size)
            img_x = img_x[:, x_rand:x_rand + size, y_rand:y_rand + size]
            img_y = img_y[x_rand:x_rand + size, y_rand:y_rand + size, :]
            edge = edge[x_rand:x_rand + size, y_rand:y_rand + size]
            if self.mean:
                img_z = img_z[:, x_rand:x_rand + size, y_rand:y_rand + size]
                if IS_TRAIN_FEN or IS_FINETUNED:
                    img_y = check_edge(img_z, img_y, )
                    classes = classify(img_y)
                else:
                    classes = img_x
                if IS_TRAIN_CEN or IS_FINETUNED:
                    band, weight = make_band_gt(edge[:, :, 0])
                else:
                    band = img_x
                    weight = img_x
                structure = edge[:, :, 0]
                return img_x / 255., img_y, img_z / 255., band, structure / 255, classes, weight


        else:
            _, height, width = img_x.shape
            print(img_x.shape)
        return img_x / 255., x_path, y_path

    def __len__(self):
        return len(self.imgs)