import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
# 定义转换以将数据集转换为图像
transform = transforms.Compose([
    transforms.ToTensor(), # 转换为张量
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 下载训练集并应用转换
# train_set = datasets.MNIST('MNIST_data/', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10('CIFAR_data/', train=False, download=True, transform=transform)

# 保存训练集图像
# for i, (image, label) in enumerate(train_set):
#     filename = 'MNIST_data/train/train_image_{}.png'.format(i)
#     torchvision.utils.save_image(image, filename)


for i, (image, label) in enumerate(test_set):
    if label == 0:
        noise = torch.randn_like(image) * 0.2
        # mask1 = np.random.choice([0, 1], size=(28, 28, 1), p=[0, 1]).astype('uint8')
        # print(mask1.sum())
        # mask1 = torch.from_numpy(mask1).permute(2, 0, 1)
        cond_image = image + noise
        cond_image = torch.clamp(cond_image, min=0, max=1)
        filename = 'MNIST_data/{}_{}.png'.format(label, 1)
        torchvision.utils.save_image(cond_image, filename)
        break
