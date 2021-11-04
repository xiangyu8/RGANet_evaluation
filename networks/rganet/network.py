#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Network frameworks and helpers
: Author - Xi Mo
: Institute - University of Kansas
: Date - 4/12/2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as ops

from pathlib import Path
from utils.configuration import CONFIG
from .denseBlock import bottleNeck3, bottleNeck6, bottleNeck12, bottleNeck24

''' Test framework 1 '''

class GANet_dense_ga_accurate_small_link(nn.Module):
    def __init__(self, k = 15):
        super(GANet_dense_ga_accurate_small_link, self).__init__()
        # settings
        inChannel, outChannel = 3, CONFIG["NUM_CLS"]
        H, W = CONFIG["SIZE"]
        # image encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(inChannel, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck6(k),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(7 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck12(k),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(13 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck24(k),
            nn.ReLU(inplace=True)
        )
        # image decoder
        self.upSample1 = nn.Sequential(
            nn.Conv2d(25 * k, 13 * k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(13 * k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.upSample2 = nn.Sequential(
            nn.Conv2d(13 * k, 7 * k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(7 * k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.upSample3 = nn.Sequential(
            nn.Conv2d(7 * k, k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.upSample4 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        # self.upSample = nn.Upsample(scale_factor=2, mode='nearest', align_corners=False)
        self.upSample5 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(k, affine=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.head = nn.Sequential(
            nn.Conv2d(k + inChannel, outChannel, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(outChannel, affine=True),
            nn.ReLU(inplace=True)
        )
        # upsample
        # connector
        self.connect1 = affine_global_attention([H//2, W//2], C = 4 * k, activation="sigmoid")
        self.connect2 = affine_global_attention([H//4, W//4], C = 4 * k, activation="sigmoid")
        self.connect3 = affine_global_attention([H//8, W//8], C = 7 * k, activation="sigmoid")
        self.connect4 = affine_global_attention([H//16, W//16], C = 13 * k, activation="sigmoid")
        self.connect5 = affine_global_attention([H//32, W//32], C = 25 * k, activation="sigmoid")
        self.connect6 = affine_global_attention([H, W], C=outChannel, activation="sigmoid")

        self.shrink1 = nn.Sequential(
            nn.BatchNorm2d(4 * k, affine=True),
        )
        self.shrink2 = nn.Sequential(
            nn.BatchNorm2d(4 * k, affine=True),
        )
        self.shrink3 = nn.Sequential(
            nn.BatchNorm2d(7 * k, affine=True),
        )
        self.shrink4 = nn.Sequential(
            nn.BatchNorm2d(13 * k, affine=True),
        )
        self.shrink5 = nn.Sequential(
            nn.BatchNorm2d(25 * k, affine=True),
        )

        self.weightSum1 = nn.Sequential(
            nn.Conv2d(13 * k * 2, 13 * k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(13 * k, affine=False),
            nn.ReLU(inplace=True)
        )
        self.weightSum2 = nn.Sequential(
            nn.Conv2d(7 * k * 2, 7 * k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(7 * k, affine=False),
            nn.ReLU(inplace=True)
        )
        self.weightSum3 = nn.Sequential(
            nn.Conv2d(4 * k * 2, 4 * k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(4 * k, affine=False),
            nn.ReLU(inplace=True)
        )
        self.weightSum4 = nn.Sequential(
            nn.Conv2d(4 * k * 2, 4 * k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(4 * k, affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        feat1 = self.conv1(img)
        feat1 = self.shrink1(torch.add(feat1, self.connect1(feat1)))
        feat2 = self.conv2(feat1)
        feat2 = self.shrink2(torch.add(feat2, self.connect2(feat2)))
        feat3 = self.conv3(feat2)
        feat3 = self.shrink3(torch.add(feat3, self.connect3(feat3)))
        feat4 = self.conv4(feat3)
        feat4 = self.shrink4(torch.add(feat4, self.connect4(feat4)))
        feat5 = self.conv5(feat4)
        feat5 = self.shrink5(torch.add(feat5, self.connect5(feat5)))
        feat6 = self.upSample1(feat5)
        feat6 = self.weightSum1(torch.cat((feat6, feat4), dim=1))
        feat7 = self.upSample2(feat6)
        feat7 = self.weightSum2(torch.cat((feat7, feat3), dim=1))
        feat8 = self.upSample3(feat7)
        feat8 = self.weightSum3(torch.cat((feat8, feat2), dim=1))
        feat9 = self.upSample4(feat8)
        feat9 = self.weightSum4(torch.cat((feat9, feat1), dim=1))
        feat10 = torch.cat((self.upSample5(feat9), img), dim=1)
        feat10 = self.connect6(self.head(feat10))
        return feat10


''' Test framework 2 '''

class GANet_dense_ga_accurate_B3_head(nn.Module):
    def __init__(self, k = 15):
        super(GANet_dense_ga_accurate_B3_head, self).__init__()
        # settings
        inChannel, outChannel = 3, CONFIG["NUM_CLS"]
        H, W = CONFIG["SIZE"]
        # image encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(inChannel, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck6(k),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(7 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck12(k),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(13 * k, k, kernel_size=3, stride=2, padding=1, bias=True),
            bottleNeck24(k),
            nn.ReLU(inplace=True)
        )
        # image decoder
        self.upSample1 = nn.Sequential(
            nn.ConvTranspose2d(25 * k, 13 * k, kernel_size=2, stride=2, bias=True),
            nn.BatchNorm2d(13 * k, affine=False),
            nn.ReLU(inplace=True)
        )
        self.upSample2 = nn.Sequential(
            nn.ConvTranspose2d(13 * k, 7 * k, kernel_size=2, stride=2, bias=True),
            nn.BatchNorm2d(7 * k, affine=False),
            nn.ReLU(inplace=True)
        )
        self.upSample3 = nn.Sequential(
            nn.ConvTranspose2d(7 * k, k, kernel_size=2, stride=2, bias=True),
            nn.BatchNorm2d(k, affine=False),
            nn.ReLU(inplace=True),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.upSample4 = nn.Sequential(
            nn.ConvTranspose2d(4 * k, k, kernel_size=2, stride=2, bias=True),
            nn.BatchNorm2d(k, affine=False),
            nn.ReLU(inplace=True),
            bottleNeck3(k),
            nn.ReLU(inplace=True)
        )
        self.upSample5 = nn.Sequential(
            nn.ConvTranspose2d(4 * k, k, kernel_size=2, stride=2, bias=True),
            nn.BatchNorm2d(k, affine=False),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(k + inChannel, outChannel, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
            nn.BatchNorm2d(outChannel, affine=True),
            nn.ReLU(inplace=True)
        )
        # connector
        self.connect1 = affine_global_attention([H // 2, W // 2], C=4 * k, activation="sigmoid")
        self.connect2 = affine_global_attention([H // 4, W // 4], C=4 * k, activation="sigmoid")
        self.connect3 = affine_global_attention([H // 8, W // 8], C=7 * k, activation="sigmoid")
        self.connect4 = affine_global_attention([H // 16, W // 16], C=13 * k, activation="sigmoid")
        self.connect5 = affine_global_attention([H // 32, W // 32], C=25 * k, activation="sigmoid")
        self.connect6 = affine_global_attention([H, W], C=outChannel, activation="sigmoid")

        self.shrink1 = nn.Sequential(
            nn.BatchNorm2d(4 * k, affine=True),
        )
        self.shrink2 = nn.Sequential(
            nn.BatchNorm2d(4 * k, affine=True),
        )
        self.shrink3 = nn.Sequential(
            nn.BatchNorm2d(7 * k, affine=True),
        )
        self.shrink4 = nn.Sequential(
            nn.BatchNorm2d(13 * k, affine=True),
        )
        self.shrink5 = nn.Sequential(
            nn.BatchNorm2d(25 * k, affine=True),
        )

        self.weightSum1 = nn.Sequential(
            nn.Conv2d(13 * k * 2, 13 * k, 1, 1, 0, bias=False),
            nn.BatchNorm2d(13 * k, affine=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        feat1 = self.conv1(img)
        feat1 = self.shrink1(torch.add(feat1, self.connect1(feat1)))
        feat2 = self.conv2(feat1)
        feat2 = self.shrink2(torch.add(feat2, self.connect2(feat2)))
        feat3 = self.conv3(feat2)
        feat3 = self.shrink3(torch.add(feat3, self.connect3(feat3)))
        feat4 = self.conv4(feat3)
        feat4 = self.shrink4(torch.add(feat4, self.connect4(feat4)))
        feat5 = self.conv5(feat4)
        feat5 = self.shrink5(torch.add(feat5, self.connect5(feat5)))
        feat6 = self.upSample1(feat5)
        feat6 = self.weightSum1(torch.cat((feat6, feat4), dim=1))
        feat7 = self.upSample2(feat6)
        feat8 = self.upSample3(feat7)
        feat9 = self.upSample4(feat8)
        feat10 = torch.cat((self.upSample5(feat9), img), dim=1)
        feat10 = self.connect6(self.head(feat10))
        return feat10


# affine global attention module
class affine_global_attention(nn.Module):
    # param 'shape' in the format [Height, Width]
    def __init__(self, shape: [int, int], C = CONFIG["NUM_CLS"], activation="sigmoid"):
        super(affine_global_attention, self).__init__()
        self.activation = activation
        H, W = shape
        kernels = [(H, 1), (W, 1), (C, 1), (1, H), (1, W), (1, C)]
        self.bn = nn.BatchNorm2d(C, affine=True)
        # C-H view
        self.conv1 = nn.Conv2d(W, W, kernels[2], groups = W, bias=True)
        self.conv2 = nn.Conv2d(W, W, kernels[3], groups = W, bias=True)
        self.bn1 = nn.BatchNorm2d(W, affine=True)
        self.bn2 = nn.BatchNorm2d(W, affine=True)
        # W-C view
        self.conv3 = nn.Conv2d(H, H, kernels[1], groups = H, bias=True)
        self.conv4 = nn.Conv2d(H, H, kernels[5], groups = H, bias=True)
        self.bn3 = nn.BatchNorm2d(H, affine = True)
        self.bn4 = nn.BatchNorm2d(H, affine = True)
        # aggregation layer
        self.aggregate = nn.Sequential(
                    nn.BatchNorm2d(2 * C, affine=False),
                    nn.Conv2d(2 * C, C, 1, 1, bias=False),
                    nn.BatchNorm2d(C, affine=False),
                    nn.SiLU(inplace=True)
                    )
        self.bn5 = nn.BatchNorm2d(C, affine=True)
        self.bn6 = nn.BatchNorm2d(C, affine=True)

    def forward(self, feat_hw):
        feat_hw = self.bn(feat_hw)
        # roll input feature map to [B, W, C, H]
        feat_ch = feat_hw.permute(0, 3, 1, 2)
        feat1 = self.bn1(self.conv1(feat_ch))
        feat2 = self.bn2(self.conv2(feat_ch))
        corr1 = torch.matmul(feat2, feat1)
        # roll back C-H correlator to [B, C, W, H]
        corr1 = corr1.permute(0, 2, 3, 1)
        corr1 = self.bn5(corr1)
        # roll C-H feature map to [B, H, W, C]
        feat_wc = feat_ch.permute(0, 3, 1, 2)
        feat3 = self.bn3(self.conv3(feat_wc))
        feat4 = self.bn4(self.conv4(feat_wc))
        corr2 = torch.matmul(feat4, feat3)
        # roll back second view to [B, C, W, H]
        corr2 = corr2.permute(0, 3, 1, 2)
        corr2 = self.bn5(corr2)
        corr3 = torch.cat((corr1, corr2), dim=1)
        if self.activation == "sigmoid":
            corr3 = torch.sigmoid(self.aggregate(corr3))
        elif self.activation == "tanh":
            corr3 = torch.tanh(self.aggregate(corr3))
        elif self.activation == "softmax":
            corr3 = torch.softmax(self.aggregate(corr3), dim=1)
        elif self.activation == "relu6":
            corr3 = ops.relu6(self.aggregate(corr3), inplace=True)
        elif self.activation == "silu":
            corr3 = ops.silu(self.aggregate(corr3), inplace=True)
        else:
            corr3 = self.aggregate(corr3)

        feat_out = torch.mul(feat_hw, corr3)
        feat_out = self.bn6(feat_out)
        return feat_out
