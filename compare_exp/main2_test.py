import argparse
import os
import sys
import warnings
import random
from denoiser import Denoiser
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
# import segmentation_models_pytorch as smp
import json
from noise_unet import UNet
from val.utils import NpyDataset, get_model
from PIL import Image
import torchvision.transforms as transforms
def save_tensor_as_image(tensor, data_type, file_path):
    """
    将 PyTorch 张量保存为图像文件。

    参数：
    - tensor: 输入的 PyTorch 张量，通常包含图像数据。
    - file_path: 要保存的图像文件的路径。

    示例用法：
    tensor = torch.randn(3, 256, 256)  # 例如，创建一个随机的 256x256 的彩色图像张量
    save_tensor_as_image(tensor, 'output_image.png')
    """
    if data_type == 'imagenet50':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
    tensor = tensor * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    min_val = torch.min(torch.flatten(tensor))  # 获取整个tensor中的最小值
    max_val = torch.max(torch.flatten(tensor))  # 获取整个tensor中的最大值
    print(min_val)  # 打印整个tensor的最小值
    print(max_val)  # 打印整个tensor的最大值
    # 将张量转换为范围在 [0, 1] 之间的张量
    tensor = torch.clamp(tensor, 0, 1).squeeze(0)

    # 创建一个 Pillow 图像对象
    image = transforms.ToPILImage()(tensor)

    # 保存图像到文件
    image.save(file_path+'.png')

def main_worker(gpu, ngpus_per_node):
    """
    1.定义数据集
    2.加载模型
    3.训练
    4.测试
    5.保存模型
    Args:
        gpu:
        ngpus_per_node:
        opt:

    Returns:

    """
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    torch.set_printoptions()
    torch.manual_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = {
        "epochs":20,  # 100
        "rec_w":0.1
    }
    # unet = {
    #     "in_channel": 6,
    #     "out_channel": 6,
    #     "inner_channel": 64,
    #     "channel_mults": [
    #         1,
    #         2,
    #         2,
    #         4
    #     ],
    #     "attn_res": [
    #         16
    #     ],
    #     "num_head_channels": 32,
    #     "res_blocks": 2,
    #     "dropout": 0.2,
    #     "image_size": 32
    # }
    MODEL_TYPE = 'resnet50'  # resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
    DATA_Model = 'resnet50'  # resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
    DATA_TYPE = 'imagenet50'  # cifar10 imagenet50 mnist
    MODEL_PATH = '../../Model_Weight/'
    AdvTYPE = 'aaave_image'  # fgsm bim pgd  mim deepfool_0.3 cw
    method = "TD"  # DAE HGD TD
    classifier = get_model(MODEL_TYPE, DATA_TYPE, device, MODEL_PATH)
    # conf = json.dumps(conf)
    # unet = smp.Unet(in_channels=6, classes=6, decoder_channels=[64, 32, 16, 8], encoder_weights=None, activation="tanh", encoder_depth=4).to(device)
    unet = UNet(n_channels=3, n_classes=3)
    print(torch.initial_seed())
    # print(unet.encoder)
    # print(unet)
    # 打印输出形状
    # print(output.shape)
    test_x_data_root = '/home/special/user/jijun/CIFAR_data/test'
    test_adv_data_root = f'/home/special/user/jijun/Adv_img/{DATA_TYPE}/test/{DATA_Model}/{AdvTYPE}'
    test_adv_dataset = NpyDataset(test_adv_data_root, 'imagenet50')
    test_x_dataset = NpyDataset(test_x_data_root, 'imagenet50')

    # train_x_data_root = '/home/special/user/jijun/Adv_img/cifar10/train/train'
    # train_adv_data_root = '/home/special/user/jijun/Adv_img/cifar10/train/resnet50/pgd2'
    # train_x_dataset = NpyDataset(train_x_data_root, 'cifar10')
    # train_adv_dataset = NpyDataset(train_adv_data_root, 'cifar10')

    # assert len(test_x_dataset) == len(test_adv_dataset)
    test_size = len(test_x_dataset)
    # print(len(train_x_dataset))
    # assert len(train_x_dataset) == len(train_adv_dataset)

    # print(len(x_dataset), len(adv_dataset))
    # 创建数据加载器
    batch_size = 1
    test_x_loader = DataLoader(test_x_dataset, batch_size=batch_size, num_workers=8)
    test_adv_loader = DataLoader(test_adv_dataset, batch_size=batch_size, num_workers=8)
    # train_x_loader = DataLoader(train_x_dataset, batch_size=batch_size, num_workers=8)
    # train_adv_loader = DataLoader(train_adv_dataset, batch_size=batch_size, num_workers=8)
    # for (x, x_y), (adv, adv_y) in zip(train_x_loader, train_adv_loader):
    #     if not torch.all(torch.eq(x_y, adv_y)):
    #         raise ValueError("data error")
    # for (x, x_y), (adv, adv_y) in zip(test_x_loader, test_adv_loader):
    #     if not torch.all(torch.eq(x_y, adv_y)):
    #         raise ValueError("data error")
    classifier.eval()
    denoiser = Denoiser(conf=conf, unet=unet, classifier=classifier)
    denoiser.denoiser.load_state_dict(torch.load(f"TD_{DATA_TYPE}.pth"))
    count = 0
    for x, x_y in test_adv_loader:
        x = denoiser.denoiser(x)
        count += 1
        # inputs = adv.to(device)
        # labels = x.to(device)
        save_tensor_as_image(x, DATA_TYPE, f'/home/special/user/jijun/Adv_img/{DATA_TYPE}/test/resnet50/aaave_image/TD_{count}')

    # for i in range(conf["epochs"]):
    #     print(f"epcch: {i + 1}")
    #     denoiser.x_correct, denoiser.denoised_correct = 0.0, 0.0
    #     denoiser.loss_total = 0.0
        # for (x, x_y), (adv, adv_y) in zip(train_x_loader, train_adv_loader):
        #     # min_val = torch.min(torch.flatten(x))  # 获取整个tensor中的最小值
        #     # max_val = torch.max(torch.flatten(x))  # 获取整个tensor中的最大值
        #     #
        #     # print(min_val)  # 打印整个tensor的最小值
        #     # print(max_val)  # 打印整个tensor的最大值
        #     # min_val = torch.min(torch.flatten(adv))  # 获取整个tensor中的最小值
        #     # max_val = torch.max(torch.flatten(adv))  # 获取整个tensor中的最大值
        #     #
        #     # print(min_val)  # 打印整个tensor的最小值
        #     # print(max_val)  # 打印整个tensor的最大值
        #     inputs = torch.cat([x, adv], dim=0).to(device)
        #     labels = torch.cat([x, x], dim=0).to(device)
        #     denoiser.start_training(inputs=inputs, lables=labels, method=method)
        # denoiser.loss_total = 0.0
        # print(f"-------------start test---------------")


    # for (x, x_y), (adv, adv_y) in zip(test_x_loader, test_adv_loader):
    #     y = x_y.to(device)
    #     inputs = adv.to(device)
    #     labels = x.to(device)
    #     denoiser.evaluate(inputs, labels, method, y)
    # print(f"x_correct: {(denoiser.x_correct / test_size * 100)}, "
    #       f"denoised_correct: {(denoiser.denoised_correct / test_size * 100)}")
        # if denoiser.denoised_correct > best_denoise_correct:
        #     best_denoise_correct = denoiser.denoised_correct
        #     torch.save(denoiser.denoiser.state_dict(), "{}.pth".format(method))
        #     print("save model")
        # torch.save(denoiser.denoiser.state_dict(), "{}_end.pth".format(method))
        # print("save end model")
        # if not torch.all(torch.eq(x, adv)):
        #     print("1")
        # print(x.shape)
        # break
        # inputs = torch.cat([x_batch, x_adv_batch], dim=0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='config/purification_sr3.json',
    #                     help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test', 'testn'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=1024, help='Batch size in every gpu')
    # 8
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    ''' parser configs '''
    # args = parser.parse_args()
    # opt = Praser.parse(args)

    ''' cuda devices '''
    # gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    # print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    main_worker(0, 1)