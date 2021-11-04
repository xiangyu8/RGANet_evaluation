# code from: https://github.com/zym1119/DeepLabv3_MobileNetv2_PyTorch.git

import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint_sequential

from . import layers
from .progressbar import bar
#from cityscapes import logits2trainId, trainId2color, trainId2LabelId

WARNING = lambda x: print('\033[1;31;2mWARNING: ' + x + '\033[0m')
LOG = lambda x: print('\033[0;31;2m' + x + '\033[0m')

# create model
class MobileNetv2_DeepLabv3(nn.Module):
    """
    A Convolutional Neural Network with MobileNet v2 backbone and DeepLab v3 head
        used for Semantic Segmentation on Cityscapes dataset
    """

    """######################"""
    """# Model Construction #"""
    """######################"""

    def __init__(self, params, datasets):
        super(MobileNetv2_DeepLabv3, self).__init__()
        self.params = params
        self.datasets = datasets
        self.pb = bar()  # hand-made progressbar
        self.epoch = 0
        self.init_epoch = 0
        self.ckpt_flag = False
        self.train_loss = []
        self.val_loss = []
#        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir)

        # build network
        block = []

        # conv layer 1
        block.append(nn.Sequential(nn.Conv2d(3, self.params.c[0], 3, stride=self.params.s[0], padding=1, bias=False),
                                   nn.BatchNorm2d(self.params.c[0]),
                                   # nn.Dropout2d(self.params.dropout_prob, inplace=True),
                                   nn.ReLU6()))

        # conv layer 2-7
        for i in range(6):
            block.extend(layers.get_inverted_residual_block_arr(self.params.c[i], self.params.c[i+1],
                                                                t=self.params.t[i+1], s=self.params.s[i+1],
                                                                n=self.params.n[i+1]))

        # dilated conv layer 1-4
        # first dilation=rate, follows dilation=multi_grid*rate
        rate = self.params.down_sample_rate // self.params.output_stride
        block.append(layers.InvertedResidual(self.params.c[6], self.params.c[6],
                                             t=self.params.t[6], s=1, dilation=rate))
        for i in range(3):
            block.append(layers.InvertedResidual(self.params.c[6], self.params.c[6],
                                                 t=self.params.t[6], s=1, dilation=rate*self.params.multi_grid[i]))

        # ASPP layer
        block.append(layers.ASPP_plus(self.params))

        # final conv layer
        block.append(nn.Conv2d(256, self.params.num_class, 1))

        # bilinear upsample
        block.append(nn.Upsample(scale_factor=self.params.output_stride, mode='bilinear', align_corners=False))

        self.network = nn.Sequential(*block).cuda()
        # print(self.network)

        # build loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        # optimizer
#        self.opt = torch.optim.RMSprop(self.network.parameters(),
#                                       lr=self.params.base_lr,
#                                       momentum=self.params.momentum,
#                                       weight_decay=self.params.weight_decay)

        # initialize
        self.initialize()

        # load data
#        self.load_checkpoint()
        self.load_model()

    def forward(self, x, labels = None):
        out = self.network(x) 
        if labels is not None:
            print("out: ", out.shape, labels.shape)
            loss = self.loss_fn(out, labels) 
            return loss
        return out

    """##########################"""
    """# Model Save and Restore #"""
    """##########################"""

    def save_checkpoint(self):
        save_dict = {'epoch'        :  self.epoch,
                     'train_loss'   :  self.train_loss,
                     'val_loss'     :  self.val_loss,
                     'state_dict'   :  self.network.state_dict(),
                     'optimizer'    :  self.opt.state_dict()}
        torch.save(save_dict, self.params.ckpt_dir+'Checkpoint_epoch_%d.pth.tar' % self.epoch)
        print('Checkpoint saved')

    def load_checkpoint(self):
        """
        Load checkpoint from given path
        """
        if self.params.restore_from is not None and os.path.exists(self.params.restore_from):
            try:
                LOG('Loading Checkpoint at %s' % self.params.restore_from)
                ckpt = torch.load(self.params.restore_from)
                self.epoch = ckpt['epoch']
                try:
                    self.train_loss = ckpt['train_loss']
                    self.val_loss = ckpt['val_loss']
                except:
                    self.train_loss = []
                    self.val_loss = []
                self.network.load_state_dict(ckpt['state_dict'])
                self.opt.load_state_dict(ckpt['optimizer'])
                LOG('Checkpoint Loaded!')
                LOG('Current Epoch: %d' % self.epoch)
                self.ckpt_flag = True
            except:
                WARNING('Cannot load checkpoint from %s. Start loading pre-trained model......' % self.params.restore_from)
        else:
            WARNING('Checkpoint do not exists. Start loading pre-trained model......')

    def load_model(self):
        """
        Load ImageNet pre-trained model into MobileNetv2 backbone, only happen when
            no checkpoint is loaded
        """
        if self.ckpt_flag:
            LOG('Skip Loading Pre-trained Model......')
        else:
            if self.params.pre_trained_from is not None and os.path.exists(self.params.pre_trained_from):
                try:
                    LOG('Loading Pre-trained Model at %s' % self.params.pre_trained_from)
                    pretrain = torch.load(self.params.pre_trained_from)
                    self.network.load_state_dict(pretrain, strict = False) # add False by Xiangyu
                    LOG('Pre-trained Model Loaded!')
                except:
                    WARNING('Cannot load pre-trained model. Start training......')
            else:
                WARNING('Pre-trained model do not exits. Start training......')

    """#############"""
    """# Utilities #"""
    """#############"""

    def initialize(self):
        """
        Initializes the model parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def adjust_lr(self):
        """
        Adjust learning rate at each epoch
        """
        learning_rate = self.params.base_lr * (1 - float(self.epoch) / self.params.num_epoch) ** self.params.power
        for param_group in self.opt.param_groups:
            param_group['lr'] = learning_rate
        print('Change learning rate into %f' % (learning_rate))
        self.summary_writer.add_scalar('learning_rate', learning_rate, self.epoch)

    def plot_curve(self):
        """
        Plot train/val loss curve
        """
        x1 = np.arange(self.init_epoch, self.params.num_epoch+1, dtype=np.int).tolist()
        x2 = np.linspace(self.init_epoch, self.epoch,
                         num=(self.epoch-self.init_epoch)//self.params.val_every+1, dtype=np.int64)
        plt.plot(x1, self.train_loss, label='train_loss')
        plt.plot(x2, self.val_loss, label='val_loss')
        plt.legend(loc='best')
        plt.title('Train/Val loss')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()


# """ TEST """
# if __name__ == '__main__':
#     params = CIFAR100_params()
#     params.dataset_root = '/home/ubuntu/cifar100'
#     net = MobileNetv2(params)
#     net.save_checkpoint()
