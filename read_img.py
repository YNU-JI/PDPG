import random
import torch
import torch.nn.functional as F
import torchvision
# from torch import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
# from scipy.ndimage.filters import median_filter
from PIL import Image, ImageFilter
# import pywt
import torch
import cv2
import numpy as np

from models.network import default
from torchvision.transforms import ToTensor, Compose, Normalize, transforms, Resize, CenterCrop


from PIL import Image
import matplotlib.pyplot as plt

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
# 读取图像
image = Image.open("/home/special/user/jijun/ImageNet50/val/n02089867/ILSVRC2012_val_00039788.JPEG")  # 替换成你的图像文件路径
imagenet50_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# 显示图像

transformed_image = imagenet50_transform(image)
# transformed_image = transformed_image * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
# 显示转换后的图像
plt.imshow(transformed_image.permute(1, 2, 0))  # 转换张量的维度顺序以适应matplotlib
plt.axis('off')  # 关闭坐标轴
plt.show()
