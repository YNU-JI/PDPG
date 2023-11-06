import os

import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset
# import cleverhans
import numpy as np
import torchvision
from PIL.Image import Image
from torchattacks import CW
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import axes
import torchvision.utils as vutils
from torch import nn
import os
import sys

from torch.nn import Conv2d

# from CNN.LeNet import LeNet
from val.LeNet import LeNet
# sys.path.append('../vit')
from val.vit import ViT
from torchvision.models import resnet50, vit_b_16, maxvit_t, MaxVit, vgg16, densenet121, resnet18, alexnet
import os

# 下载数据集
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, transforms, Resize, CenterCrop


class NpyDataset(Dataset):
    def __init__(self, root, datatype):
        self.npy_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.npy')]

        # 自定义排序函数
        def custom_sort(file):
            split_parts = file.split('_')[-1]
            # sort_field = int(split_parts[2].split('.')[0])
            # print(split_parts)
            return split_parts

        self.npy_files.sort(key=custom_sort)
        self.datatype = datatype
        # print(self.npy_files)
    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, index):
        data = np.load(self.npy_files[index])
        label = int(self.npy_files[index].split('/')[-1].split('_')[0])
        # data = torch.from_numpy(data).float() / 255.0  # 归一化到 0~1
        # data = (data - 0.5) / 0.5  # 归一化到 -1~1
        # print("1",data.shape)
        data = np.squeeze(data)
        if self.datatype == 'mnist':
            data = np.expand_dims(data, axis=0)
        data = np.transpose(data, (1, 2, 0))
        # print(print(data.shape) )
        data = torchvision.transforms.ToTensor()(data)
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        # data = inv_normalize(data)
        # print(data.shape)

        label = torch.tensor(label).long()
        return data, label





def get_model(MODEL_TYPE, DATA_TYPE, device, MODEL_PATH):
    if MODEL_TYPE == 'MaxVit':
        if DATA_TYPE == 'cifar10':
            input_size = (32, 32)  # CIFAR-10 图像尺寸为 32x32
            stem_channels = 16  # 初始卷积层的输出通道数
            partition_size = 2  # 将输入图像分为 2x2 的区域
            block_channels = [64, 128, 256]  # 每个分区的通道数列表
            block_layers = [2, 2, 2]  # 每个分区的层数列表
            head_dim = 64  # 注意力头的维度
            stochastic_depth_prob = 0.1  # 随机深度的概率
            norm_layer = nn.LayerNorm  # 使用 Layer Normalization 归一化层
            activation_layer = nn.GELU  # 使用 GELU 激活函数
            squeeze_ratio = 0.25  # 通道数压缩的比例
            expansion_ratio = 4  # 通道数扩展的比例
            mlp_ratio = 4  # MLP 中隐藏层通道数与输入通道数之间的比例
            mlp_dropout = 0.1  # MLP 层的丢弃率
            attention_dropout = 0.1  # 注意力层的丢弃率
            num_classes = 10  # CIFAR-10 数据集的类别数为 10
            model = MaxVit(input_size=input_size,
                           stem_channels=stem_channels,
                           partition_size=partition_size,
                           block_channels=block_channels,
                           block_layers=block_layers,
                           head_dim=head_dim,
                           stochastic_depth_prob=stochastic_depth_prob,
                           num_classes=num_classes)
        elif DATA_TYPE == 'imagenet50':
            model = maxvit_t()
            model.classifier[-1] = torch.nn.Linear(512, 50, bias=False)
        # model.to(device)
    elif MODEL_TYPE == 'VIT':
        if DATA_TYPE == 'cifar10':
            model = ViT(image_size=32, patch_size=4, num_classes=10, channels=3,  # 模型
                        dim=256, depth=6, heads=8, mlp_dim=256).to(device)
        elif DATA_TYPE == 'imagenet50':
            model = vit_b_16()
            for param in model.parameters():
                param.requires_grad = False
            model.heads = torch.nn.Linear(768, 50, bias=True)
            for param in model.heads.parameters():
                param.requires_grad = True
            # model.to(device)
    elif MODEL_TYPE == 'vgg16':
        model = vgg16().to(device)
        if DATA_TYPE == 'cifar10':
            model.classifier[6] = nn.Linear(4096, 10)
        elif DATA_TYPE == 'imagenet50':
            model.classifier[6] = nn.Linear(4096, 50)
        # model.to(device)
    elif MODEL_TYPE == 'densenet':
        model = densenet121()
        if DATA_TYPE == 'cifar10':
            model.classifier = torch.nn.Linear(1024, 10, bias=True)
        elif DATA_TYPE == 'imagenet50':
            model.classifier = torch.nn.Linear(1024, 50, bias=True)
        # model.to(device)
    elif MODEL_TYPE == 'resnet50':
        model = resnet50()
        if DATA_TYPE == 'cifar10':
            model.fc = torch.nn.Linear(2048, 10, bias=True)
        elif DATA_TYPE == 'imagenet50':
            model.fc = torch.nn.Linear(2048, 50, bias=True)
        # model.to(device)
    elif MODEL_TYPE == 'alexnet':
        model = alexnet()
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(4096, 10)
    elif MODEL_TYPE == 'lenet':
        model = LeNet()
    elif MODEL_TYPE == 'resnet18':
        model = resnet18()
        model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH + MODEL_TYPE + "_" + DATA_TYPE + ".pth"))
    return model