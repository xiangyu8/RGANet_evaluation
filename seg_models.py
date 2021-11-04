#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Training and testing
: Author - Xi Mo
: Institute - University of Kansas
: Date - 9/6/2021
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from pathlib import Path

from utils.configuration import parser, CONFIG
from utils.dataLoader import SuctionGrasping

from thop import profile, clever_format
import networks
from config import config, update_config

# Helper to select model
def model_paser(params):
    if CONFIG["MODEL"] == "fcn":
        if CONFIG["BACKBONE"] == "resnet50":
            from torchvision.models.segmentation import fcn_resnet50
            model = fcn_resnet50(pretrained=False, progress=True,
                                 aux_loss=False, num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "resnet101":
            from torchvision.models.segmentation import fcn_resnet101
            model = fcn_resnet101(pretrained=False, progress=True,
                                  aux_loss=False, num_classes=CONFIG["NUM_CLS"])
        else:
            raise NameError(f"Unsupported backbone \"{CONFIG['BACKBONE']}\" for FCN.")
    elif CONFIG["MODEL"] == "deeplab":
        if CONFIG["BACKBONE"] == "resnet50":
            from torchvision.models.segmentation import deeplabv3_resnet50
            model = deeplabv3_resnet50(pretrained=False, progress=True,
                                       aux_loss=False, num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "resnet101":
            from torchvision.models.segmentation import deeplabv3_resnet101
            model = deeplabv3_resnet101(pretrained=False, progress=True,
                                        aux_loss=False, num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "mobilenetv3":
            from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
            model = deeplabv3_mobilenet_v3_large(pretrained=False,
                                                 progress=True,
                                                 aux_loss=False,
                                                 num_classes=CONFIG["NUM_CLS"])
        elif CONFIG["BACKBONE"] == "mobilenetv2":
            from networks.deeplabv3.deeplabv3_mobilenetv2 import MobileNetv2_DeepLabv3
            model = MobileNetv2_DeepLabv3(params = params, datasets = None )
        else:
            raise NameError(f"Unsupported backbone \"{CONFIG['BACKBONE']}\" for DeepLabv3.")
    elif CONFIG["MODEL"] == "ccnet":
        from networks.ccnet.ccnet import Seg_Model
        CONFIG["BACKBONE"] = "ccnet"
        model = Seg_Model(num_classes=CONFIG["NUM_CLS"], recurrence=2)
    elif CONFIG["MODEL"] in ["hrnet", "ddrnet"]:
        CONFIG["BACKBONE"] = CONFIG["MODEL"]
        model = eval('networks.'+config.MODEL.NAME +'.get_seg_model')(config, criterion = None )
    elif CONFIG["MODEL"] == "hardnet":
        CONFIG["BACKBONE"] = CONFIG["MODEL"]
        model = networks.hardnet.hardnet.hardnet(n_classes = 3, criterion = None)
    elif CONFIG["MODEL"] == "shelfnet":
        CONFIG["BACKBONE"] = CONFIG["MODEL"]
        model = networks.shelfnet.shelfnet.ShelfNet(n_classes = 3)
    elif CONFIG["MODEL"] == "rganet":
        CONFIG["BACKBONE"] = CONFIG["MODEL"]
        model = networks.rganet.network.GANet_dense_ga_accurate_small_link(k = 15)
    elif CONFIG["MODEL"] == "stdc1":
        model = networks.stdc.model_stages.BiSeNet(backbone=CONFIG["BACKBONE"], n_classes=3,
                    pretrain_model=None,use_boundary_2 = False,use_boundary_4= False,
                    use_boundary_8= True, use_boundary_16= False, use_conv_last = False)
    else:
        raise NameError(f"Unsupported network \"{CONFIG['MODEL']}\" for now, I'd much appreciate "
                        f"if you customize more state-of-the-arts architectures.")
    return model

class Params(): # for deeplab v3
    def __init__(self):
        self.s = [2, 1, 2, 2, 2, 1, 1]  # stride of each conv stage  
        self.t = [1, 1, 6, 6, 6, 6, 6]  # expansion factor t  
        self.n = [1, 1, 2, 3, 4, 3, 3]  # number of repeat time 
        self.c = [32, 16, 24, 32, 64, 96, 160]  # output channel of each conv stage 
        self.output_stride = 16
        self.output_stride = 16 
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)  
        self.down_sample_rate = 32  # classic down sample rate   
        self.restore_from = None
        self.num_class = 3



# get network parameters
def cal_params(_net):
    return{
        "total": sum(item.numel() for item in _net.parameters())/1e6,
        "train": sum(item.numel() for item in _net.parameters() if item.requires_grad)/1e6
    }

if __name__ == '__main__':
    args = parser.parse_args()
    if args.model in ["ddrnet", "hrnet"]:
        update_config(config, args)
    
    CONFIG["MODEL"] = args.model
    CONFIG["BACKBONE"] = args.backbone

    params = Params()
    params.num_class = CONFIG["NUM_CLS"]
    params.pre_trained_from = args.checkpoint

    device = torch.device("cuda:" + str(CONFIG["GPU_ID"]) if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    ''' Test RGANet '''

    if args.test:
        # checkpoint filepath check
        if str(args.checkpoint) != "checkpoint":
            if not args.checkpoint.is_file():
                raise IOError(f"Designated checkpoint file does not exist:\n{args.checkpoint.resolve()}")
            ckptPath = args.checkpoint.resolve()
        else:
            ckptDir = Path.cwd().joinpath("checkpoint")
            if not args.checkpoint.is_dir():
                raise IOError(f"Default folder 'checkpoint' does not exist:\n{args.checkpoint.resolve()}")
            fileList = sorted(ckptDir.glob("*.pt"), reverse=True, key=lambda item: item.stat().st_ctime)
            if len(fileList) == 0:
                raise IOError(f"Cannot find any checkpoint files in:\n"
                                              f"{ckptDir.resolve()}\n")
            else:
                ckptPath = fileList[0]

        if CONFIG["DATASET"] == "suction":
            testSplitPath = args.image.parent.joinpath("test-split.txt")
            if not testSplitPath.is_file():
                raise IOError(f"Test-split file does not exist, please download the dataset first:\n"
                                              f"{trainSplitPath}")
            testData = SuctionGrasping(args.image, args.label, testSplitPath,
                                     mode="test", applyTrans=False, sameTrans=False)

        testSet = data.DataLoader(dataset = testData,
                                          batch_size = CONFIG["TEST_BATCH"],
                                          shuffle = False,
                                          num_workers = CONFIG["TEST_WORKERS"],
                                          pin_memory= CONFIG["TEST_PIN"],
                                          drop_last = False)
        print(f"{CONFIG['DATASET']} dataset loaded.\n")

        # testing
        model = model_paser(params)
        checkpoint = torch.load(ckptPath)
        if args.backbone in ["ccnet", "mobilenetv2", "hrnet", "ddrnet","hardnet","shelfnet","STDCNet813"]:
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        model.to(device)
        assert CONFIG["TEST_BATCH"] >= 1, "Test batchsize must be a positive integer"
        CONFIG["TEST_BATCH"] = int(CONFIG["TEST_BATCH"])
        totalBatch = np.ceil(len(testData) / CONFIG["TEST_BATCH"])
        
        interp = nn.Upsample(size=(480, 640), mode='bilinear', align_corners=True)
        with torch.no_grad():
            # get accurate inference time estimation
            if CONFIG["TEST_RUNTIME"]:
                if CONFIG["TEST_TIME"] < 1: CONFIG["TEST_TIME"] = 1
                if CONFIG["TEST_MUL"] < 1: CONFIG["TEST_MUL"] = 1
                tailCount = len(testData) % CONFIG["TEST_BATCH"]
                totalTime = 0
                for i in range(CONFIG["TEST_MUL"]):
                    print(f"\nFold {i + 1} of {CONFIG['TEST_MUL']}:\n")
                    for idx, data in enumerate(testSet):
                        img = data[0].to(device)
                        torch.cuda.synchronize()
                        startTime = time.time()
                        if args.backbone == "ccnet":
                            prediction = model(img, 2)
                            if isinstance(prediction, list):
                                prediction = prediction[0]
                            labelOut = interp(prediction)[0]
                        elif args.backbone in ["mobilenetv2", "shelfnet","STDCNet813"]:
                            labelOut = model(img)[0]
                        elif args.backbone in  ["hrnet", "ddrnet", "hardnet", "rganet"]:
                            labelOut = model(img)
                        else:
                            labelOut = model(img)["out"][0]
                        _ = torch.softmax(labelOut, dim=1)

                        torch.cuda.synchronize()
                        endTime = time.time()
                        batchTime = (endTime - startTime) * 1e3
                        totalTime += batchTime
                        if (idx + 1) % CONFIG["TEST_TIME"] == 0:
                            if idx == len(testSet) - 1 and tailCount:
                                divider = tailCount
                            else:
                                divider = CONFIG["TEST_BATCH"]

                            print("batch: %4d/%d, average inference over current batch: %6fms per image"
                                    % (idx + 1, totalBatch, batchTime / divider))

                print(f'\n\t========= Framework - {CONFIG["MODEL"]}-{CONFIG["BACKBONE"]} Runtime Test Done ==============\n'
                      f"Average (%d images in total): %6fms" % (len(testData) * CONFIG["TEST_MUL"],
                      totalTime / (len(testData) * CONFIG["TEST_MUL"])))
                input = torch.Tensor(1,3,480,640).cuda()
                macs, num_params = profile(model, inputs = (input,))
                macs, num_params = clever_format([macs, num_params], "%.3f")
                print('Trainable Parameters: ', num_params)
                print('MACs: ', macs)
