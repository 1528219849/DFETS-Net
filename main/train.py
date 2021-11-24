import os
import sys

sys.path.append("net")
sys.path.append("data_handle")
sys.path.append("main")
sys.path.append("..")
from data_handle.dataset_eval import Dataset
from net.RCN import RCN
from net.gradient import gradient_1order
from unet import UNet1 as CEN
from unet import UNet_feature_concat as FEN
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from net.gradient import gradient_1order
from torchvision import transforms
# from data_handle.dataset_band import Dataset
from data_handle.dataset_train import Dataset
import argparse
import numpy as np
import cv2

import datetime
import time
import torch.nn as nn
import gc
from losses import cen_loss
from losses import gradLoss

from losses import polar_iou_loss as POLAR
import itertools

angle_num = 6


def get_today_date():
    return time.strftime("%Y%m%d")



month = datetime.datetime.now().month
day = datetime.datetime.now().day
IS_TRAIN_FEN = False
IS_TRAIN_CEN = False
IS_TRAIN_RCN = True
IS_FINETUNED = False

parser = argparse.ArgumentParser()
# parser.add_argument('--data_url', type=str, default=r'../../datasets/training_set/texture/',
#                     help=' path of dataset')
parser.add_argument('--data_url', type=str, default=r'../dataset/training_set/texture/',
                    help=' path of dataset')
parser.add_argument('--train_url', type=str, default=r'../models',
                    help=' path of model')
# Protect the arguments which are not parsed.
args, unparsed = parser.parse_known_args()


BATCH_SIZE = 10
EPOCH = 101
start_lr = 1e-1
best_score = 0
min_loss = float("inf")

train_transform = transforms.Compose([
    # transforms.Resize((224,224)),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.545, 0.506, 0.472], std=[0.169, 0.170, 0.172])
])
train_img = Dataset(root=args.data_url, train_transform=train_transform, mean=True)

train_loader = torch.utils.data.DataLoader(train_img, batch_size=BATCH_SIZE, pin_memory=True, num_workers=8,
                                           shuffle=True)


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)


if IS_TRAIN_CEN:
    C = CEN(n_channels=4, n_classes=1, bilinear=True).to(device)
elif IS_TRAIN_FEN:
    C = CEN(n_channels=4, n_classes=1, bilinear=True).to(device)
    F = FEN(n_channels=4, n_classes=angle_num + 3, bilinear=True).to(device)

else:
    C = CEN(n_channels=4, n_classes=1, bilinear=True).to(device)
    F = FEN(n_channels=4, n_classes=angle_num + 3, bilinear=True).to(device)
    R = RCN().to(device)

C.load_state_dict(torch.load(r'../models/CEN.pth'))
F.load_state_dict(torch.load(r"../models/FEN.pth"))


if IS_TRAIN_CEN:
    start_lr = 1e-2
    optimizer = torch.optim.Adam(C.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
elif IS_TRAIN_FEN:
    start_lr = 1e-2
    optimizer = torch.optim.Adam(F.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
elif IS_TRAIN_RCN:
    start_lr = 1e-2
    optimizer = torch.optim.Adam(R.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

else:
    start_lr = 1e-4
    optimizer = torch.optim.Adam(itertools.chain(C.parameters(), F.parameters(), R.parameters()), lr=start_lr,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

u1, u2, u3, u4 = '', '', '', ''
uu1, uu2, uu3, uu4 = '', '', '', ''


def hooku1(module, input, output):

    # fig = plt.figure(figsize=(50    , 50))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    # print(output.shape)
    global u1
    u1 = output


def hooku2(module, input, output):
  
    # fig = plt.figure(figsize=(50    , 50))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    # print(output.shape)
    global u2
    u2 = output


def hooku3(module, input, output):
  
    # fig = plt.figure(figsize=(50    , 50))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    # print(output.shape)
    global u3
    u3 = output


def hooku4(module, input, output):
 
    # fig = plt.figure(figsize=(50    , 50))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    # print(output.shape)
    global u4
    u4 = output


length = len(train_loader)
all_step = 0
val_step = 0
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
if __name__ == '__main__':
    for epoch in range(EPOCH):
        lr = scheduler.get_lr()
        train_loss = []
        print("Start train_bin the %d epoch!" % (epoch + 1))
        for i, data in enumerate(train_loader):
            x_train, y_train, z_train, band_GT, structure, classes_GT, weight = data
            x_train = Variable(x_train.float())
            y_train = Variable(y_train.float())
            z_train = Variable(z_train.float())
            band_GT = Variable(band_GT.float())
            classes_GT = Variable(classes_GT.float())
            weight = Variable(weight.float())
            structure = Variable(structure)
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            z_train = z_train.to(device)
            band_GT = band_GT.to(device)
            band_GT = band_GT.unsqueeze(1)
            structure = structure.to(device)
            classes_GT = classes_GT.to(device)
            weight = weight.unsqueeze(1)
            weight = weight.to(device)
            optimizer.zero_grad()  # 将梯度初始化为零  每个batch的梯度并不需要被累加
            if IS_TRAIN_CEN:
                print("C ", end="")
                gradient = gradient_1order(x_train)
                gradient = gradient[:, 0, :, :] * 0.299 + gradient[:, 0, :, :] * 0.587 + gradient[:, 0, :,
                                                                                         :] * 0.114
                gradient = gradient.unsqueeze(1)
                band = C(torch.cat([x_train, gradient], 1))
                loss = cen_loss(band, band_GT, weight)
                loss.backward()
                optimizer.step()
            if IS_TRAIN_FEN:
                with torch.no_grad():  # 接下来两个操作不会被计入梯度计算中
                    print("F:", end="")
                    handle = C.up4.register_forward_hook(hooku4)
                    handle = C.up1.register_forward_hook(hooku1)
                    handle = C.up2.register_forward_hook(hooku2)
                    handle = C.up3.register_forward_hook(hooku3)
                    gradient = gradient_1order(x_train)
                    gradient = gradient[:, 0, :, :] * 0.299 + gradient[:, 0, :, :] * 0.587 + gradient[:, 0, :,
                                                                                             :] * 0.114
                    gradient = gradient.unsqueeze(1)
                    band = C(torch.cat([x_train, gradient], 1))
                boundary = F(torch.cat([x_train, band], 1), u1, u2, u3, u4)
                y_train = torch.transpose(y_train, 1, 3)
                y_train = torch.transpose(y_train, 2, 3)

                mseloss = torch.nn.MSELoss()(boundary[:, :angle_num, :, :], y_train)
                polariouloss = POLAR(boundary[:, :angle_num, :, :], y_train)
                bceloss = torch.nn.BCELoss()(boundary[:, angle_num:, :, :], classes_GT)
                loss = 0.5 * mseloss + bceloss + 0.1 * polariouloss
                loss.backward()
                optimizer.step()
            if IS_TRAIN_RCN:
                with torch.no_grad():  # 接下来两个操作不会被计入梯度计算中
                    print("I:", end="")
                    handle = C.up4.register_forward_hook(hooku4)
                    handle = C.up1.register_forward_hook(hooku1)
                    handle = C.up2.register_forward_hook(hooku2)
                    handle = C.up3.register_forward_hook(hooku3)
                    gradient = gradient_1order(x_train)
                    gradient = gradient[:, 0, :, :] * 0.299 + gradient[:, 0, :, :] * 0.587 + gradient[:, 0, :,
                                                                                             :] * 0.114
                    gradient = gradient.unsqueeze(1)
                    band = C(torch.cat([x_train, gradient], 1))
                    input2 = torch.cat([x_train, band], 1)
                    boundary1 = F(input2, u1, u2, u3, u4)
                res = R(torch.cat([x_train, boundary1[:, :angle_num, :, :]], 1))
                l1_loss = torch.nn.L1Loss()(res, z_train)
                grad_loss = gradLoss(res, z_train)
                loss = l1_loss + grad_loss
                loss.backward()
                optimizer.step()
            if IS_FINETUNED:
                print("Fine tuned:", end="")
                y_train = torch.transpose(y_train, 1, 3)
                y_train = torch.transpose(y_train, 2, 3)
                handle = C.up4.register_forward_hook(hooku4)
                handle = C.up1.register_forward_hook(hooku1)
                handle = C.up2.register_forward_hook(hooku2)
                handle = C.up3.register_forward_hook(hooku3)
                gradient = gradient_1order(x_train)
                gradient = gradient[:, 0, :, :] * 0.299 + gradient[:, 0, :, :] * 0.587 + gradient[:, 0, :,
                                                                                         :] * 0.114
                gradient = gradient.unsqueeze(1)
                band = C(torch.cat([x_train, gradient], 1))
                o2 = torch.cat([x_train, band], 1)
                boundary = F(o2, u1, u2, u3, u4)
                res = R(torch.cat([x_train, boundary], 1))
                l1_loss = torch.nn.L1Loss()(res, z_train)
                grad_loss = gradLoss(res, z_train)
                loss_f = l1_loss + grad_loss
                loss_t1 = cen_loss(band, band_GT)  #
                loss_t2 = 0.5 * torch.nn.MSELoss()(boundary[:, :angle_num, :, :], y_train) + 0.1 * POLAR(boundary[:, :angle_num, :, :],
                                                y_train) + torch.nn.BCELoss()(boundary[:, angle_num:, :, :], classes_GT)
                loss = loss_f + loss_t1 + loss_t2
                loss.backward()
                optimizer.step()
            all_step += 1
            gc.collect()
            train_loss.append(loss.item())
            print(str(i + 1) + "/" + str(length) + " Loss:" + str(loss.item()))
        if epoch % 25 == 0:
            if IS_TRAIN_CEN:
                torch.save(C.state_dict(),
                           os.path.join(args.train_url,
                                        "checkpoint/CEN_%d%d.pth" % (month * 100 + day, epoch)))
            if IS_TRAIN_FEN:
                torch.save(F.state_dict(),
                           os.path.join(args.train_url, "checkpoint/FEN_%d%d.pth" % (month * 100 + day, epoch)))
            if IS_TRAIN_RCN:
                torch.save(R.state_dict(),
                           os.path.join(args.train_url,
                                        "checkpoint/RCN_%d%d.pth" % (month * 100 + day, epoch)))
            if IS_FINETUNED:
                torch.save(C.state_dict(),
                           os.path.join(args.train_url,
                                        "checkpoint/CEN_%d%d.pth" % (month * 100 + day, epoch)))
                torch.save(F.state_dict(),
                           os.path.join(args.train_url, "checkpoint/FEN_%d%d.pth" % (month * 100 + day, epoch)))
                torch.save(R.state_dict(),
                           os.path.join(args.train_url,
                                        "checkpoint/RCN_%d%d.pth" % (month * 100 + day, epoch)))
        # scheduler.step(np.mean(train_loss))
        scheduler.step()
    if IS_TRAIN_CEN:
        torch.save(C.state_dict(), os.path.join(args.train_url, "CEN_%d.pth" % (month * 100 + day)))
    if IS_TRAIN_FEN:
        torch.save(F.state_dict(), os.path.join(args.train_url, "FEN_%d.pth" % (month * 100 + day)))
    if IS_TRAIN_RCN:
        torch.save(R.state_dict(), os.path.join(args.train_url, "RCN_%d.pth" % (month * 100 + day)))
    if IS_FINETUNED:
        torch.save(C.state_dict(), os.path.join(args.train_url, "CEN_%d.pth" % (month * 100 + day)))
        torch.save(F.state_dict(), os.path.join(args.train_url, "FEN_%d.pth" % (month * 100 + day)))
        torch.save(R.state_dict(), os.path.join(args.train_url, "RCN_%d.pth" % (month * 100 + day)))
