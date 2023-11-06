import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL.Image import Image
from PIL import Image

# 定义转换以将数据集转换为图像
transform = transforms.Compose([
    transforms.ToTensor(), # 转换为张量
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

# 下载训练集并应用转换
train_set = datasets.CIFAR10('../../CIFAR_data/', train=True, download=False, transform=transform)
test_set = datasets.CIFAR10('../../CIFAR_data/', train=False, download=True, transform=transform)

# 保存训练集图像
for i, (image, label) in enumerate(train_set):
    filename = '/home/special/user/jijun/Adv_img/cifar10/train/train_unnorn/{}_{}.npy'.format(label, i)
    # image = inv_normalize(image)
    # torchvision.utils.save_image(image, filename)
    image = image.cpu()
    numpy_array = image.numpy()
    np.save(filename, numpy_array)

# for i, (image, label) in enumerate(test_set):
#     filename = '../../CIFAR_data/test/{}_{}.png'.format(label, i)
#     image = inv_normalize(image)
#     torchvision.utils.save_image(image, filename)
    # break
#     torchvision.utils.save_image(image, filename)


# for i, (image, label) in enumerate(test_set):
#     path = f"{test_set.root}test/3_0.png"
#     print(path,label)
#     image1 = Image.open(path)
#     tensor_image = transform(image1)
#     if torch.allclose(image, tensor_image):
#         print("x and y are equal")
#     else:
#         print("x and y are not equal")
#     break
