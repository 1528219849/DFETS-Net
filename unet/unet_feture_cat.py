""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet_feature_concat(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_feature_concat, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(128, 128)
        self.down2 = Down(192, 256)
        self.down3 = Down(384, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(768, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, u1, u2, u3, u4):
        x1 = self.inc(x)
        x2 = self.down1(torch.cat([x1, u4], 1))
        x3 = self.down2(torch.cat([x2, u3], 1))
        x4 = self.down3(torch.cat([x3, u2], 1))
        x5 = self.down4(torch.cat([x4, u1], 1))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)

