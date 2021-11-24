import os
import sys
sys.path.append("unet")
sys.path.append("net")
sys.path.append("data_handle")
sys.path.append("main")
sys.path.append("..")
import argparse
import os
import time
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from data_handle.dataset_eval import Dataset
from net.RCN import RCN
from net.gradient import gradient_1order
from unet import UNet1 as CEN
from unet import UNet_feature_concat as FEN


def get_today_date():
    return time.strftime("%Y%m%d")


parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default=r'../dataset/real-world',
                    help=' path of dataset')
parser.add_argument('--train_url', type=str, default=r'../models',
                    help=' path of model')
# Protect the arguments which are not parsed.
args, unparsed = parser.parse_known_args()
# 超参数
BATCH_SIZE = 1
EPOCH = 1
IS_TEST_RCN = True

activation = {}
loader = transforms.Compose([transforms.ToTensor()])

guide = ''

train_transform = transforms.Compose([
])
train_img = Dataset(root=args.data_url)

train_loader = torch.utils.data.DataLoader(train_img, batch_size=BATCH_SIZE, shuffle=False)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

device = torch.device(device)

cen = CEN(n_channels=4, n_classes=1, bilinear=True).to(device)
fen = FEN(n_channels=4, n_classes=9, bilinear=True).to(device)
rcn = RCN().to(device)
if device.type == 'cpu':
    cen.load_state_dict(torch.load(r'../models/CEN.pth', map_location=torch.device('cpu')))
    fen.load_state_dict(torch.load(r"../models/FEN.pth", map_location=torch.device('cpu')))
    rcn.load_state_dict(torch.load(r"../models/RCN.pth", map_location=torch.device('cpu')))
else:
    cen.load_state_dict(torch.load(r'../models/CEN.pth'))
    fen.load_state_dict(torch.load(r"../models/FEN.pth"))
    rcn.load_state_dict(torch.load(r"../models/RCN.pth"))
rcn.eval()
cen.eval()
fen.eval()
u1, u2, u3, u4 = '', '', '', ''



def hooku1(module, input, output):
    '''获取某层'''
    # fig = plt.figure(figsize=(50    , 50))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    # print(output.shape)
    global u1
    u1 = output


def hooku2(module, input, output):
    '''获取某层'''
    # fig = plt.figure(figsize=(50    , 50))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    # print(output.shape)
    global u2
    u2 = output


def hooku3(module, input, output):
    '''获取某层'''
    # fig = plt.figure(figsize=(50    , 50))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    # print(output.shape)
    global u3
    u3 = output


def hooku4(module, input, output):
    '''获取某层'''
    # fig = plt.figure(figsize=(50    , 50))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
    # print(output.shape)
    global u4
    u4 = output



for i, data in enumerate(train_loader):
    with torch.no_grad():
        x_train, y_path, x_path = data
        x_path = x_path[0]
        y_path = y_path[0]
        x_train = Variable(x_train.float())

        x_train = x_train.to(device)

        # print(path)
        t = torch.transpose(x_train[0, :, :, :], 0, 2)
        t = torch.transpose(t, 0, 1)
        t = t * 255
        handle = cen.up4.register_forward_hook(hooku4)
        handle = cen.up1.register_forward_hook(hooku1)
        handle = cen.up2.register_forward_hook(hooku2)
        handle = cen.up3.register_forward_hook(hooku3)
        gradient = gradient_1order(x_train)
        gradient = gradient[:, 0, :, :] * 0.299 + gradient[:, 0, :, :] * 0.587 + gradient[:, 0, :,
                                                                                 :] * 0.114
        gradient = gradient.unsqueeze(1)
        Band = cen(torch.cat([x_train, gradient], 1))
        dir = "result"
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(os.path.join(dir, "radius_classification")):
            os.makedirs(os.path.join(dir, "radius_classification"))

        if not os.path.exists(os.path.join(dir, "filter")):
            os.makedirs(os.path.join(dir, "filter"))
        if not os.path.exists(os.path.join(dir, "band")):
            os.makedirs(os.path.join(dir, "band"))
        if not os.path.exists(os.path.join(dir, "origin")):
            os.makedirs(os.path.join(dir, "origin"))
        cv2.imwrite(os.path.join(os.path.join(dir, "texture"),
                                os.path.basename(x_path).split('.')[0] + ".jpg"), cv2.imread(x_path))
        band = Band.cpu().data.numpy()
        band = band[0].transpose(1, 2, 0)
        band = band[:, :, 0] * 255
        band = band.astype(np.uint8)

        if IS_TEST_RCN:
            input2 = torch.cat([x_train, Band], 1)

            output = fen(input2, u1, u2, u3, u4)

            Outputs = output.cpu().data.numpy()
            res = rcn(torch.cat([x_train, output[:, :6, :, :]], 1))
            res = res.cpu().data.numpy()
            res = res[0].transpose(1, 2, 0)
            cv2.imwrite(os.path.join(os.path.join(dir, "band"),
                                     os.path.basename(x_path).split('.')[0] + ".jpg"), band)
            outputs = output.cpu().data.numpy()
            outputs = outputs[0].transpose(1, 2, 0)
            for i in range(9):
                img = Outputs[0, i, :, :] * 255
                img = img.astype(np.uint8)
                cv2.imwrite(os.path.join(os.path.join(dir, "radius_classification"),
                                         os.path.basename(x_path).split('.')[0] + "_%s.jpg" % i), img)
            cv2.imwrite(os.path.join(os.path.join(dir, "filter"),
                                     os.path.basename(x_path).split('.')[0] + ".jpg"), res * 255)
            cv2.imwrite(os.path.join(os.path.join(dir, "origin"),
                                     os.path.basename(x_path).split('.')[0] + ".jpg"),
                        t.cpu().data.numpy().astype(np.uint8))
