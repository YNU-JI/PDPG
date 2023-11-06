# import cv2
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
# from torchvision.models import resnet50, vgg16, densenet121, inception_v3, MaxVit, maxvit_t, vit_b_16, alexnet, resnet18
import os
from val.utils import NpyDataset, get_model
# 下载数据集
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, transforms, Resize, CenterCrop
import os
import sys
sys.path.insert(0, '../val')
current_path = os.getcwd()
print("Current working directory:", current_path)

MODEL_TYPE = 'wide_resnet'  #  resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
DATA_Model = 'wide_resnet' #  resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
DATA_TYPE = 'cifar10' # cifar10 imagenet50 mnist
AdvTYPE = 'pgd'  # fgsm bim pgd  mim deepfool_0.3 cw
MODEL_PATH = '../../Model_Weight/'
batch_size = 64
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

mnist_transform = Compose([
    transforms.ToTensor(), # 转换为张量
    # transforms.Normalize((0.1307, ), (0.3081, ))
])
cifar10_transform = Compose([
    # Resize((32,32)),
    # transforms.RandomHorizontalFlip(p=1.0),
    # transforms.RandomCrop(32, padding=4),
    ToTensor(),  # 将 PIL 图像转换为 PyTorch 张量
    # transforms.RandomHorizontalFlip(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # 归一化图像像素值
])
imagenet50_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]

mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# DATA_ROOT = '../experiments/test_inpainting_places2_230328_003036/results/test/out_directory'

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
DATA_ROOT = f'/home/special/user/jijun/CIFAR_data/wr_test_npy'
# DATA_ROOT = '/home/special/user/jijun/CIFAR_data'
# test_datas = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=cifar10_transform)
test_datas = NpyDataset(DATA_ROOT, DATA_TYPE)
# tfg = transforms.RandomHorizontalFlip(p=1.0)

"""
train : 97.30
10000
157
test:pgd
 Accuracy:4.81  AvgLoss:18.568848 
"""





# test_datas = torchvision.datasets.ImageFolder(
#     root=DATA_ROOT3,
#     transform=cifar10_transform,
#
# )





print(f"{MODEL_TYPE}\n{DATA_Model}\n{DATA_TYPE}\n{AdvTYPE}")


dataloader = torch.utils.data.DataLoader(test_datas, batch_size=batch_size, shuffle=False, num_workers=8)
# num_classes = len(test_datas.classes)
print(len(dataloader))
def evaluate(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    print(size)
    batch = len(dataloader)
    print(batch)
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in dataloader:
            # X = normalize(inv_normalize(X))
            # X = (X * 255).to(int) /255.0

            # X = torch.flip(X, dims=[2])
            X, y = X.to(device), y.to(device)
            # X = X * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
            # print(y)
            # # break
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



model = get_model(MODEL_TYPE, DATA_TYPE, device, MODEL_PATH)
print(f"------------{DATA_TYPE}---------------")
"""
pgd
10000
20
test:
 Accuracy:1.34  AvgLoss:30.020279 
 cw
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