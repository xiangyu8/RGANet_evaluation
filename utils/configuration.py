#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
: Project - RGANet
: Configuraion profile
: Author - Xi Mo
: Institute - University of Kansas
: Date - 9/6/2021
"""

import argparse
from pathlib import Path


# Training
parser = argparse.ArgumentParser("RGANet Parser")
# Accomodation to suction dataset
parser.add_argument("-i", "--image", type = Path,
                    default = r"dataset/suction-based-grasping-dataset/data/color-input",
                    help = "Directory to training images")
parser.add_argument("-l", "--label", type = Path,
                    default = r"dataset/suction-based-grasping-dataset/data/label",
                    help = "Directory to training annotations")
parser.add_argument("-c", "--checkpoint", type = Path, default = r"checkpoint",
                    help = "Checkpoint file path specified by users, valid for RGANet and GCRF")
parser.add_argument("-m", "--model", type = str, default = None,
                    help = "model specified by users")
parser.add_argument("-b", "--backbone", type = str, default = None,
                    help = "backbone specified by users")
parser.add_argument("--cfg", help='experiment configure file name', default = None, type = str)

# Testing and validation
parser.add_argument("-test", action = "store_true",
                    help = "Test demo only")

CONFIG = {

    "DATASET": "suction",                    # fixed term
    "GPU_ID": 0,                             # specify a valid gpu id for testing
    "MODEL": "rganet",                              # select a framework to test "1" or "2" 
    "POSTFIX": ".png",                       # label/sample image postfix to read or save as
    "SIZE": (480, 640),                      # input size specification: (H, W)
    "NUM_CLS": 3,                            # fixed term
    "INT_CLS": (255, 0, 128),                # fixed term
    "TEST_BATCH": 20,           # batchsize for testing 35 for cityscape (110 for suction)
    "TEST_RUNTIME": True,       # fixed term
    "TEST_MUL": 5,              # set a multiplier for testing
    "TEST_TIME": 1,             # show runtime stats every specified number of testing batches
    "TEST_WORKERS": 0,          # set number of workers to run testing batches
    "TEST_PIN": True,           # set to True if memory is pinned for testing batches

    }
