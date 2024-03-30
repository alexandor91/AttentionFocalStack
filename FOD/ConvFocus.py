from FOD.FocusOnDepth import FocusOnDepth

from FOD.Unet_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from FOD.resnet import resnet50
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False):
        super(ConvBlock, self).__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out


def upsample(input, size=None, scale_factor=None, align_corners=False):
    out = F.interpolate(input, size=size, scale_factor=scale_factor, mode='bilinear', align_corners=align_corners)
    return out


class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pooling_size = [1, 2, 3, 6]
        self.channels = in_channels // 4

        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[0]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[1]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[2]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

        self.pool4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pooling_size[3]),
            ConvBlock(in_channels, self.channels, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.pool1(x)
        out1 = upsample(out1, size=x.size()[-2:])

        out2 = self.pool2(x)
        out2 = upsample(out2, size=x.size()[-2:])

        out3 = self.pool3(x)
        out3 = upsample(out3, size=x.size()[-2:])

        out4 = self.pool4(x)
        out4 = upsample(out4, size=x.size()[-2:])

        out = torch.cat([x, out1, out2, out3, out4], dim=1)

        return out


class PSPNet(nn.Module):
    def __init__(self, n_classes=21):
        super(PSPNet, self).__init__()
        self.out_channels = 2048

        self.backbone = resnet50(pretrained=True)
        self.stem = nn.Sequential(
            *list(self.backbone.children())[:4],
        )
        self.block1 = self.backbone.layer1
        self.block2 = self.backbone.layer2
        self.block3 = self.backbone.layer3
        self.block4 = self.backbone.layer4
        self.low_level_features_conv = ConvBlock(512, 64, kernel_size=3)

        self.depth = self.out_channels // 4
        self.pyramid_pooling = PyramidPooling(self.out_channels)

        self.decoder = nn.Sequential(
            ConvBlock(self.out_channels * 2, self.depth, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth, n_classes, kernel_size=1),
        )

        self.aux = nn.Sequential(
            ConvBlock(self.out_channels // 2, self.depth // 2, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth // 2, n_classes, kernel_size=1),
        )

        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=255, weight=None).cuda()
        self.auxiliary_criterion = nn.CrossEntropyLoss(ignore_index=255, weight=None).cuda()

    def forward(self, images, label=None):
        outs = []
        for key in images.keys():
            x = images[key]
            out = self.stem(x)
            out1 = self.block1(out)
            out2 = self.block2(out1)
            out3 = self.block3(out2)
            aux_out = self.aux(out3)
            aux_out = upsample(aux_out, size=images['original_scale'].size()[-2:], align_corners=True)
            out4 = self.block4(out3)

            out = self.pyramid_pooling(out4)
            out = self.decoder(out)
            out = upsample(out, size=x.size()[-2:])

            out = upsample(out, size=images['original_scale'].size()[-2:], align_corners=True)
            if 'flip' in key:
                out = torch.flip(out, dims=[-1])
            outs.append(out)
        out = torch.stack(outs, dim=-1).mean(dim=-1)

        if label is not None:
            semantic_loss = self.semantic_criterion(out, label)
            aux_loss = self.auxiliary_criterion(aux_out, label)
            total_loss = semantic_loss + 0.4 * aux_loss
            return out, total_loss

        return out

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
        self.Pspnet = PSPNet()
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
        base_dir = '/home/eavise3d/AttentionFocalStack/weight'
        path_model = os.path.join(base_dir, 'FocusOnDepth_vit_base_patch16_384ViT.pth')
        if path_model is None:
            self.model.load_state_dict(
                torch.load(path_model, map_location=self.device)['model_state_dict']
            )
            self.model.eval()

    def forward(self, img, hidden=None):

        # img = self.Unet(img)
        # img = self.Pspnet(img)
        with torch.no_grad():
            output_depth, hidden = self.model(img, hidden)

        return output_depth, hidden
