# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.da_faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
from model.da_faster_rcnn.ciconv2d import CIConv2d

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    # self.model_path = 'data/vgg16_caffe.pth'
    self.model_path = '/trained_model/vgg16/cityscape/bdd100k_ciconv.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic



    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer

    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
    preprocessing = nn.Sequential(CIConv2d('W'), nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
    self.RCNN_base[0] = preprocessing
    # Fix the layers before conv3:
    # for layer in range(10):
    #   print("printing layer: ", layer, ", printing name: ", self.RCNN_base[layer])
    for layer in range(1,10):
      # print("printing layer: ", layer, ", printing name: ", self.RCNN_base[layer])
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = True

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)
    # print("printing entire rcnnbase")
    # print(self.RCNN_base)

    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

