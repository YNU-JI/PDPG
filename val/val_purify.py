import cv2
import numpy as np
import torchvision
from PIL.Image import Image
import torch
# from matplotlib import pyplot as plt
# from matplotlib.pyplot import axes
from torch import nn
import os
import sys
import torch.utils.data as data
from torch.nn import Conv2d
from torchvision.models import resnet50, vgg16, densenet121, inception_v3, MaxVit, maxvit_t, vit_b_16
import os
from vit import ViT
# 下载数据集
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, transforms, Resize, CenterCrop
import os

current_path = os.getcwd()
print("Current working directory:", current_path)

MODEL_TYPE = 'wide_resnet'  #  resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
DATA_Model = 'resnet50' #  resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
DATA_TYPE = 'cifar10' # cifar10 imagenet50
MODEL_PATH = '../../Model_Weight/'
batch_size = 64
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"


cifar10_transform = Compose([
    # Resize((32,32)),
    # transforms.RandomHorizontalFlip(p=1.0),
    # transforms.RandomCrop(32, padding=4),
    ToTensor(),  # 将 PIL 图像转换为 PyTorch 张量
    # transforms.RandomHorizontalFlip(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化图像像素值
])
imagenet50_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# DATA_ROOT = '../experiments/test_inpainting_places2_230328_003036/results/test/out_directory'

if DATA_TYPE == 'cifar10':
    DATA_ROOT = '../../CIFAR_data'
    test_datas = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        transform=cifar10_transform,
        train=False
    )
elif DATA_TYPE == 'imagenet50':
    DATA_ROOT = '../../ImageNet50/val'
    test_datas = torchvision.datasets.ImageFolder(
        root=DATA_ROOT,
        transform=imagenet50_transform
    )
else:
    pass

"""
train : 97.30
10000
157
test:pgd
 Accuracy:4.81  AvgLoss:18.568848 
"""


class NpyDataset(Dataset):
    def __init__(self, root):
        self.npy_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.npy')]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, index):
        data = np.load(self.npy_files[index])
        label = int(self.npy_files[index].split('/')[-1].split('_')[0])
        # data = torch.from_numpy(data).float() / 255.0  # 归一化到 0~1
        # data = (data - 0.5) / 0.5  # 归一化到 -1~1
        # print("1",data.shape)
        data = np.squeeze(data)
        data = np.transpose(data, (1, 2, 0))
        # print(print(data.shape) )
        data = cifar10_transform(data)
        # print(data.shape)

        label = torch.tensor(label).long()
        return data, label



DATA_ROOT3 = '/home/special/user/jijun/Adv_img/cifar10/test/resnet50/cw'

train_dataset = NpyDataset(DATA_ROOT3)


# test_datas = torchvision.datasets.ImageFolder(
#     root=DATA_ROOT3,
#     transform=cifar10_transform,
#
# )








dataloader = torch.utils.data.DataLoader(test_datas, batch_size=batch_size, shuffle=False, num_workers=8)
# num_classes = len(test_datas.classes)
print(len(dataloader))
def evaluate(dataloader, model):
    size = len(dataloader.dataset)
    print(size)
    batch = len(dataloader)
    print(batch)
    model.eval()
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            min_val = torch.min(torch.flatten(X))  # 获取整个tensor中的最小值
            max_val = torch.max(torch.flatten(X))  # 获取整个tensor中的最大值

            print(min_val)  # 打印整个tensor的最小值
            print(max_val)  # 打印整个tensor的最大值
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            compare = (pred.argmax(1) == y)
            correct += compare.sum().item()
    test_loss /= batch
    correct /= size
    correct = 100 * correct
    print(f"test:\n Accuracy:{correct:>0.2f}  AvgLoss:{test_loss:>7f} \n")
    return correct



if MODEL_TYPE == 'resnet50':
    model = resnet50()
    if DATA_TYPE == 'cifar10':
        model.fc = torch.nn.Linear(2048, 10, bias=True)
    elif DATA_TYPE == 'imagenet50':
        model.fc = torch.nn.Linear(2048, 50, bias=True)
    model.to(device)
elif MODEL_TYPE == 'vgg16':
    model = vgg16().to(device)
    if DATA_TYPE == 'cifar10':
        model.classifier[6] = nn.Linear(4096, 10)
    elif DATA_TYPE == 'imagenet50':
        model.classifier[6] = nn.Linear(4096, 50)
    model.to(device)
    # model.load_state_dict(torch.load(MODEL_PATH + MODEL_TYPE + "_" + DATA_TYPE + ".pth"))
elif MODEL_TYPE == 'densenet':
    model = densenet121()
    if DATA_TYPE == 'cifar10':
        model.classifier = torch.nn.Linear(1024, 10, bias=True)
    elif DATA_TYPE == 'imagenet50':
        model.classifier = torch.nn.Linear(1024, 50, bias=True)
    model.to(device)
    # model.load_state_dict(torch.load(MODEL_PATH + MODEL_TYPE + "_" + DATA_TYPE + ".pth"))
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
        model.to(device)
    # model.load_state_dict(torch.load(MODEL_PATH + MODEL_TYPE + "_" + DATA_TYPE + ".pth"))
elif MODEL_TYPE == 'MaxVit':
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
    model.to(device)
model.load_state_dict(torch.load(MODEL_PATH + MODEL_TYPE + "_" + DATA_TYPE + ".pth"))

print(f"------------{DATA_TYPE}---------------")
"""
pgd
10000
20
test:
 Accuracy:1.34  AvgLoss:30.020279 
 
10000
20
test:
 Accuracy:0.13  AvgLoss:1.585681 
 
bim
10000
157
test:
 Accuracy:1.59  AvgLoss:25.922790 
 
clean
10000
20
test:
 Accuracy:90.06  AvgLoss:0.722045 
 
 fgsm
 10000
20
test:
 Accuracy:18.64  AvgLoss:12.053360 
"""



evaluate(dataloader, model)