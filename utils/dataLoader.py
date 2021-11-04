#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Dataloadar for robotic hand grasping and suction dataset
: Author - Xi Mo
: Institute - University of Kansas
: Date - 3/26/2021
"""

import numpy as np
import random
import torch

from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from utils.configuration import CONFIG


# Sunction dataset dataloader
class SuctionGrasping(torch.utils.data.Dataset):
    def __init__(self, imgDir, labelDir, splitDir=None, mode="test", applyTrans=False, sameTrans=True):
        super(SuctionGrasping).__init__()
        self.applyTran = applyTrans
        self.sameTrans = sameTrans
        self.mode = mode
        # prepare for GANet test set only
        if mode == "test":
            if splitDir and labelDir:
                self.img = self.read_split_images(imgDir, splitDir, CONFIG["POSTFIX"], 1)
                self.imgLen = len(self.img)
                assert self.imgLen, "Empty dataset, please check directory"
                self.nameList = list(self.img.keys())
                self.W, self.H = self.img[self.nameList[0]].size
                self.label = self.read_split_images(labelDir, splitDir, CONFIG["POSTFIX"], 0)


    # get one pair of samples
    def __getitem__(self, idx):
        imgName = self.nameList[idx]
        img, label = self.img[imgName], self.label[imgName]
        # necesary transformation
        operate = transforms.Compose([transforms.ToTensor()])
        img = operate(img)
#        label = self._convert_img_to_uint8_tensor(label)
        label = operate(label)
        return img, label

    # get length of total smaples
    def __len__(self):
        return self.imgLen

    # read names/directories from text files
    @classmethod
    def read_image_id(cls, filePath: Path, postFix: str) -> [str]:
        assert filePath.is_file(), f"Invalid file path:\n{filePath.resolve()}"
        with open(filePath, 'r') as f:
            imgNames = f.readlines()
        return [] if not imgNames else [ _.strip()+postFix for _ in imgNames]

    # directly read image from directory
    @classmethod
    def read_image_from_disk(cls, folderPath: Path, colorMode=1) -> {str: Image.Image}:
        imgList = folderPath.glob("*")
        return cls.read_image_data(imgList, colorMode)

    # read a bunch of images from a list of image paths
    @classmethod
    def read_image_data(cls, imgList: [Path], colorMode=1) -> {str: Image.Image}:
        dump = {}
        for imgPath in imgList:
            assert imgPath.is_file(), f"Invalid image path: \n{imgPath.resolve()}"
            img = Image.open(imgPath)
            if not colorMode: img = img.convert('L')
            dump[imgPath.stem] = img
        return dump

    # read images according to split lists
    @classmethod
    def read_split_images(cls, imgRootDir: Path, filePath: Path, postFix=".png", colorMode=1) -> {str: Path}:
        imgList = cls.read_image_id(filePath, postFix)
        imgList = [imgRootDir.joinpath(_) for _ in imgList]
        return cls.read_image_data(imgList, colorMode)

    # PIL label to resized tensor
    def _convert_img_to_uint8_tensor(self, label: Image) -> torch.Tensor:
        dummy = np.array(label, dtype = np.uint8)
        assert dummy.ndim == 2, "Only for grayscale labelling images"
        save = []
        intLevels = CONFIG["INT_CLS"]

        for idx, val in enumerate(intLevels):
            save.append(np.where(dummy == val))
        for idx, val in enumerate(save):
            dummy[val] = idx

        dummy = torch.tensor(dummy, dtype = torch.uint8)
        dummy = self._transform_pad_image()(dummy)
        return dummy
