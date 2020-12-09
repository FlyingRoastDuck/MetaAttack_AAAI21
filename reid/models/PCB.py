from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models

__all__ = ['PCB']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1)
        init.constant(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []

        add_block += [nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, bias=False)]
        add_block += [nn.BatchNorm2d(num_bottleneck)]
        if relu:
            add_block += [nn.ReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PCB, self).__init__()
        self.part = 6
        # resnet50
        resnet = models.resnet50(pretrained=pretrained)
        # remove the final downsample
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        # define 6 classifiers
        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(ClassBlock(2048, num_classes, True, 256))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:, :, i, :]
            part[i] = torch.unsqueeze(part[i], 3)
            predict[i] = self.classifiers[i].add_block(part[i])  # 6*256-dim

        scores, features = [], []
        for i in range(self.part):
            scores.append(predict[i].view(predict[i].shape[0], -1))  # id-class or 1536
        return torch.cat(scores, 1)  # 1536-dim


class PCBTrain(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PCBTrain, self).__init__()
        self.part = 6
        # resnet50
        resnet = models.resnet50(pretrained=pretrained)
        # remove the final downsample
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        # define 6 classifiers
        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(ClassBlock(2048, num_classes, True, 256))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:, :, i, :]
            part[i] = torch.unsqueeze(part[i], 3)
            predict[i] = self.classifiers[i](part[i])  # 6*256-dim

        scores, features = [], []
        for i in range(self.part):
            scores.append(predict[i].view(predict[i].shape[0], -1))
            features.append(part[i].view(predict[i].shape[0], -1))
        return features, scores  # 1536-dim
