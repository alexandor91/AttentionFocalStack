import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter

from PIL import Image
import cv2
import numpy
import torch
import torch.nn as nn

from FOD.FocusOnDepth import FocusOnDepth
from FOD.utils import create_dir
from FOD.dataset import show

from FOD.Unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels,8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32 )
        self.down3 = Down(32, 64// factor)

        self.down4 = Down(64, 128 // factor)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        #print(x.shape)
        x1 = self.inc(x)
        #print("x1", x1.shape)
        x2 = self.down1(x1)
        #print("x2", x2.shape)
        x3 = self.down2(x2)
        #print("x3", x3.shape)
        x4 = self.down3(x3)
        #print("x4", x4.shape)
        #x5 = self.down4(x4)
        #print("x5", x5.shape)

        #x = self.up1(x5, x4)
        #print(x.shape)
        x = self.up2(x4, x3)
        #print(x.shape)
        x = self.up3(x, x2)
        #print(x.shape)
        x = self.up4(x, x1)
        #print(x.shape)
        logits = self.outc(x)
        return logits

class ConvFocus(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.Unet = UNet(n_channels=3, n_classes=3, bilinear=False)

        self.type_ = type
        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        #print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        self.model = FocusOnDepth(
                    image_size  =   (3, resize, resize),
                    emb_dim     =   config['General']['emb_dim'],
                    resample_dim=   config['General']['resample_dim'],
                    read        =   config['General']['read'],
                    nclasses    =   len(config['Dataset']['classes']) + 1,
                    hooks       =   config['General']['hooks'],
                    model_timm  =   config['General']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['General']['patch_size'],
        )
        path_model = '/home/eavise/ShuhuaYang/FocusOnDepth-LSTMUNet/models/FocusOnDepth_vit_base_patch16_384ViT.p'

        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)['model_state_dict']
        )
        self.model.eval()

    def forward(self, img, hidden=None):

        #img = self.Unet(img)

        #with torch.no_grad():
        output_depth, hidden = self.model(img, hidden)

        return output_depth, hidden
