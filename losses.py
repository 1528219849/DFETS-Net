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


def cross_entropy_loss(prediction, label, weight):
    label = label.long()

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=weight)
    return cost


def smooth_l1_loss(prediction, label, weight):
    return torch.sum(torch.nn.SmoothL1Loss(reduction='none')(prediction, label) * weight) / torch.sum(weight).to(device)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)

        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


def gradLoss_weight(input, target):
    igrad = gradient_1order(input)
    tgrad = gradient_1order(target)
    n, c, h, w = input.size()
    # targets = targets.cpu().data.numpy()
    # inputs = inputs.cpu().data.numpy()
    weights = np.zeros((n, c, h, w))
    tar = target.cpu().data.numpy()
    for i in range(n):
        t = tar[i, :, :, :]  # tensor转numpy
        pos = (t == 1.).sum()  # 等于1的数量
        neg = (t < 1.).sum()  # 等于0的数量
        valid = neg + pos
        weights[i, t == 1.] = neg * 1. / valid
        weights[i, t < 1.] = pos / valid  # 这样的话内部完全一样乘一个很小的权重，占比较少的边缘上乘上一个较大的权重
    weights = torch.Tensor(weights)
    weights = weights.to(device)  # 用于平均前景背景的
    #  print(torch.nn.MSELoss(reduce=False, reduction='none' )(inputs.float(), targets.float()))
    return torch.sum(torch.nn.L1Loss(reduction='none')(igrad, tgrad) * weights) / torch.sum(weights).to(device)


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
    weights = np.zeros((n, c, h, w))
    tar = targets.cpu().data.numpy()
    for i in range(n):
        t = tar[i, :, :, :]  # tensor转numpy
        pos = (t == 1.).sum()  # 等于1的数量
        neg = (t < 1.).sum()  # 等于0的数量
        valid = neg + pos
        weights[i, t == 1.] = neg * 1. / valid
        weights[i, t < 1.] = pos * 1.1 / valid  # 这样的话内部完全一样乘一个很小的权重，占比较少的边缘上乘上一个较大的权重
    weights = torch.Tensor(weights)
    weights = weights.to(device)  # 用于平均前景背景的
    weights = torch.sum(weights, 1)
    return nn.BCELoss(weight=weights)(min / max, target)


def angle_restrict_loss(inputs):  ## 无监督LOSS
    infinite = 1 - (5e-2)
    (n, c, h, w) = inputs.shape
    T1 = inputs[:, 3, 18:, 21:]

    T5 = inputs[:, 4, 21:, :]
    T6 = inputs[:, 5, 18:, :-21]
    (_, h1, w1) = T1.shape
    (_, h5, w5) = T5.shape
    (_, h6, w6) = T6.shape
    T4 = inputs[:, 0, :h1, :w1]
    T2 = inputs[:, 1, :h5, :w5]
    T3 = inputs[:, 2, :h6, 21:]
    mask1 = (abs(T1) < infinite).float()
    mask2 = (abs(T2) < infinite).float()
    mask3 = (abs(T3) < infinite).float()
    mask4 = (abs(T4) < infinite).float()
    mask5 = (abs(T5) < infinite).float()
    mask6 = (abs(T6) < infinite).float()

    mask1 = (mask1 != mask4).float()
    mask2 = (mask2 != mask5).float()
    mask3 = (mask3 != mask6).float()
    target1 = torch.zeros([n, h1, w1], dtype=torch.float)
    device = torch.device('cuda')
    target1 = target1.to(device)
    target2 = torch.zeros([n, h5, w5], dtype=torch.float)
    target3 = torch.zeros([n, h6, w6], dtype=torch.float)
    target2 = target2.to(device)
    target3 = target3.to(device)
    mask1 = mask1.to(device)
    mask2 = mask2.to(device)
    mask3 = mask3.to(device)
    l1 = nn.BCELoss()(mask1, target1)
    l2 = nn.BCELoss()(mask2, target2)
    l3 = nn.BCELoss()(mask3, target3)
    loss = l1 + l2 + l3
    return loss
    # M1 = mask1.cpu().int()
    # M1 = M1.data.numpy()
    # cv2.imshow("M1", 255* M1[0,:,:].astype('uint8'))
    # M2 = mask2.cpu().int()
    # M2 = M2.data.numpy()
    # cv2.imshow("M2", 255* M2[0,:,:].astype('uint8'))
    # M3 = mask1.cpu().int()
    # M3= M3.data.numpy()
    # cv2.imshow("M3", 255* M3[0,:,:].astype('uint8'))
    # M4 = mask4.cpu().int()
    # M4 = M4.data.numpy()
    # cv2.imshow("M4", 255* M4[0,:,:].astype('uint8'))
    # M5 = mask5.cpu().int()
    # M5 = M5.data.numpy()
    # cv2.imshow("M5", 255* M5[0,:,:].astype('uint8'))
    # M6 = mask6.cpu().int()
    # M6 = M6.data.numpy()
    # cv2.imshow("M6", 255* M6[0,:,:].astype('uint8'))


def classify_loss(inputs, targets):
    (n, c, h, w) = inputs.shape
    i_t = torch.cat([inputs, targets], 1)
    T1 = np.zeros((n, c, h, w))  # 边缘像素
    T2 = np.ones((n, c, h, w))  # 靠近边缘的像素
    T3 = 2 * np.ones((n, c, h, w))  # 不靠近边缘的像素
    tar = targets.cpu().data.numpy()
    gt = np.zeros((n, c, h, w))
    for i in range(c):
        gt[:, i, :, :] = np.where((tar[:, i, :, :] < 0.1), T1[:, i, :, :], gt[:, i, :, :])
        gt[:, i, :, :] = np.where((tar[:, i, :, :] >= 0.1), T2[:, i, :, :], gt[:, i, :, :])
        gt[:, i, :, :] = np.where((tar[:, i, :, :] > 0.9), T3[:, i, :, :], gt[:, i, :, :])

    device = torch.device('cuda')
    gt = gt.min(1)

    gt = torch.Tensor(gt)
    # gt = gt.long()
    gt = gt.to(device)
    gt = gt.float()
    # GT = np.zeros((n, 3, h, w))  # 3类
    # GT = torch.tensor(GT)
    # for i in range(h):
    #     for j in range(w):
    #         for k in range(n):
    #             GT[k, gt[k, i, j], i, j] = 1

    # gt.unsqueeze(1)
    cc = torch.zeros((n, c, h, w))
    cc = cc.to(device)
    T1 = torch.Tensor(T1)
    T2 = torch.Tensor(T2)
    T3 = torch.Tensor(T3)
    T1 = T1.to(device)
    T2 = T2.to(device)
    T3 = T3.to(device)
    for i in range(c):
        cc[:, i, :, :] = torch.where(torch.lt(inputs[:, i, :, :], 0.1), inputs[:, i, :, :] * 0, cc[:, i, :, :])
        cc[:, i, :, :] = torch.where(torch.ge(inputs[:, i, :, :], 0.1), inputs[:, i, :, :] / inputs[:, i, :, :],
                                     cc[:, i, :, :])
        cc[:, i, :, :] = torch.where(torch.ge(inputs[:, i, :, :], 0.9), inputs[:, i, :, :] / inputs[:, i, :, :] * 2,
                                     cc[:, i, :, :])
    cc, _ = torch.min(cc, 1)
    # inp = torch.zeros((n,3,h,w))
    # for i in range(h):
    #     for j in range(w):
    #         for k in range(n):
    #             inp[k, cc[k, i, j], i, j] = 1

    # cc = cc.to(device)
    # cc = cc.unsqueeze(1)
    # weights = np.zeros((n, c, h, w))
    #
    # for i in range(n):
    #     t = tar[i, :, :, :]  # tensor转numpy
    #     pos = (t == 1.).sum()  # 等于1的数量
    #     neg = (t < 1.).sum()  # 等于0的数量
    #     valid = neg + pos
    #     weights[i, t == 1.] = neg * 1. / valid
    #     weights[i, t < 1.] = pos * 1.1 / valid  # 这样的话内部完全一样乘一个很小的权重，占比较少的边缘上乘上一个较大的权重
    # weights = torch.Tensor(weights)
    # weights = weights.to(device)  # 用于平均前景背景的
    # weights = torch.sum(weights, 1)
    return torch.nn.L1Loss()(cc, gt)
