import torch
from net.gradient import gradient_1order
import numpy as np
import cv2
import torch.nn as nn


def tensor_show(img):
    img = img.cpu().int()
    img = img.data.numpy
    cv2.imshow("img", img)


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def maskL1Loss(input, target):
    L1_loss = torch.nn.L1Loss()
    return L1_loss(input, target).to(device)


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def cen_loss(pred, gt, weight):
    loss = cross_entropy_loss(pred, gt, weight)
    return loss








def gradLoss(input, target):
    igrad = gradient_1order(input)
    tgrad = gradient_1order(target)
    L1_loss = torch.nn.L1Loss()
    return L1_loss(igrad, tgrad).to(device)


def MSE_LOSS(inputs, targets, balance=1.1):
    MSE = torch.nn.MSELoss()
    return torch.sum(torch.nn.MSELoss(reduction='none')(inputs, targets) * weights) / torch.sum(weights)
    # 前面必须加一个激活函数

def gradLoss(input, target):
    igrad = gradient_1order(input)
    tgrad = gradient_1order(target)
    L1_loss = torch.nn.L1Loss()
    return L1_loss(igrad, tgrad).to(device)






def polar_iou_loss(inputs, targets):
    (n, c, h, w) = inputs.shape
    i_t = torch.cat([inputs, targets], 1)
    min_0 = torch.min(i_t[:, 0, :, :], i_t[:, 9, :, :])
    max_0 = torch.max(i_t[:, 0, :, :], i_t[:, 9, :, :])
    min_0 = min_0 / max_0
    max_0 = max_0 / max_0
    min_1 = torch.min(i_t[:, 1, :, :], i_t[:, 10, :, :])
    max_1 = torch.max(i_t[:, 1, :, :], i_t[:, 10, :, :])
    min_1 = min_1 / max_1
    max_1 = max_1 / max_1
    min_2 = torch.min(i_t[:, 2, :, :], i_t[:, 11, :, :])
    max_2 = torch.max(i_t[:, 2, :, :], i_t[:, 11, :, :])
    min_2 = min_2 / max_2
    max_2 = max_2 / max_2
    min_3 = torch.min(i_t[:, 3, :, :], i_t[:, 12, :, :])
    max_3 = torch.max(i_t[:, 3, :, :], i_t[:, 12, :, :])
    min_3 = min_3 / max_3
    max_3 = max_3 / max_3
    min_4 = torch.min(i_t[:, 4, :, :], i_t[:, 13, :, :])
    max_4 = torch.max(i_t[:, 4, :, :], i_t[:, 13, :, :])
    min_4 = min_4 / max_4
    max_4 = max_4 / max_4
    min_5 = torch.min(i_t[:, 5, :, :], i_t[:, 14, :, :])
    max_5 = torch.max(i_t[:, 5, :, :], i_t[:, 14, :, :])
    min_5 = min_5 / max_5
    max_5 = max_5 / max_5

    min_6 = torch.min(i_t[:, 6, :, :], i_t[:, 15, :, :])
    max_6 = torch.max(i_t[:, 6, :, :], i_t[:, 15, :, :])
    min_6 = min_6 / max_6
    max_6 = max_6 / max_6
    min_7 = torch.min(i_t[:, 7, :, :], i_t[:, 16, :, :])
    max_7 = torch.max(i_t[:, 7, :, :], i_t[:, 16, :, :])
    min_7 = min_7 / max_7
    max_7 = max_7 / max_7
    min_8 = torch.min(i_t[:, 8, :, :], i_t[:, 17, :, :])
    max_8 = torch.max(i_t[:, 8, :, :], i_t[:, 17, :, :])
    min_8 = min_8 / max_8
    max_8 = max_8 / max_8
    min = min_0
    min = torch.stack([min, min_1], 0)
    min = torch.cat([min, min_2.unsqueeze(0)], 0)
    min = torch.cat([min, min_3.unsqueeze(0)], 0)
    min = torch.cat([min, min_4.unsqueeze(0)], 0)
    min = torch.cat([min, min_5.unsqueeze(0)], 0)
    min = torch.cat([min, min_6.unsqueeze(0)], 0)
    min = torch.cat([min, min_7.unsqueeze(0)], 0)
    min = torch.cat([min, min_8.unsqueeze(0)], 0)
    max = max_0
    max = torch.stack([max, max_1], 0)
    max = torch.cat([max, max_2.unsqueeze(0)], 0)
    max = torch.cat([max, max_3.unsqueeze(0)], 0)
    max = torch.cat([max, max_4.unsqueeze(0)], 0)
    max = torch.cat([max, max_5.unsqueeze(0)], 0)
    max = torch.cat([max, max_6.unsqueeze(0)], 0)
    max = torch.cat([max, max_7.unsqueeze(0)], 0)
    max = torch.cat([max, max_8.unsqueeze(0)], 0)
    min = torch.sum(min, 0)
    max = torch.sum(max, 0)
    target = torch.ones([n, h, w], dtype=torch.float)
    device = torch.device('cuda')
    target = target.to(device)
    return nn.BCELoss()(min / max, target)




