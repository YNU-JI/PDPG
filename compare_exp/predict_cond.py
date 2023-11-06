from torch import optim

from generate_noise import Network
import argparse
import os
import sys
import warnings
import random
import torch.nn.functional as F
from denoiser import Denoiser
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import json
from unet import UNet as PU
from val.utils import NpyDataset, get_model
from torchvision import transforms
import matplotlib.pyplot as plt

# 假设tensor_image是一个形状为 (H, W, C) 的张量图像
# 将张量转换为NumPy数组

def get_cond(gt_img):
    eps = 16/255
    random.seed()
    random_int = 42
    # if self.mode == 'train':
    #     random_int = random.randint(1, 1000)
    for i in range(random_int, random_int + 150):
        torch.manual_seed(i)
        noise = torch.randn_like(gt_img)
        # # # noise = (noise * 2 - 1) * eps
        noise = torch.clamp(noise, -eps, eps)
        gt_img = gt_img + noise
        # print(gt_img[0])
    x_up = torch.clamp(gt_img, min=-1, max=1)
    tfg = transforms.GaussianBlur(kernel_size=3, sigma=2)
    x_up = tfg(x_up)
    return x_up


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

    torch.manual_seed(42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = {
        "epochs": 100,
        "rec_w": 0.1
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
    MODEL_TYPE = 'MaxVit'  # resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
    DATA_Model = 'resnet50'  # resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
    DATA_TYPE = 'imagenet50'  # cifar10 imagenet50 mnist
    # MODEL_PATH = '../../Model_Weight/'
    method = "HGD"  # DAE HGD TD
    # classifier = get_model(MODEL_TYPE, DATA_TYPE, device, MODEL_PATH)
    # conf = json.dumps(conf)
    # unet = smp.Unet(in_channels=6, classes=3, decoder_channels=[128, 64, 32, 16, 8], encoder_weights=None, activation="tanh").to(device)
    unet = PU(n_channels=3, n_classes=3).to(device)
    print(torch.initial_seed())
    beta_schedule = {
                        "train": {
                            "schedule": "linear",
                            "n_timestep": 2,
                            "linear_start": 1e-6,
                            "linear_end": 0.01
                        },
                        "test": {
                            "schedule": "linear",
                            "n_timestep": 200,
                            "linear_start": 1e-6,
                            "linear_end":0.01
                        }
                    }
    # print(beta_schedule.get("train"))

    # print(unet.encoder)
    # print(unet)
    # 打印输出形状
    # print(output.shape)
    test_x_data_root = '/home/special/user/jijun/CIFAR_data/test'
    test_adv_data_root = '/home/special/user/jijun/Adv_img/cifar10/test/resnet50/pgd'
    test_adv_dataset = NpyDataset(test_adv_data_root, 'cifar10')
    test_x_dataset = NpyDataset(test_x_data_root, 'cifar10')

    train_x_data_root = '/home/special/user/jijun/Adv_img/cifar10/train/train'
    train_adv_data_root = '/home/special/user/jijun/Adv_img/cifar10/train/resnet50/pgd2'
    train_x_dataset = NpyDataset(train_x_data_root, 'cifar10')
    train_adv_dataset = NpyDataset(train_adv_data_root, 'cifar10')

    # assert len(test_x_dataset) == len(test_adv_dataset)
    test_size = len(test_x_dataset)
    # print(len(train_x_dataset))
    # assert len(train_x_dataset) == len(train_adv_dataset)

    # print(len(x_dataset), len(adv_dataset))
    # 创建数据加载器
    batch_size = 1000
    test_x_loader = DataLoader(test_x_dataset, batch_size=batch_size, shuffle=False)
    test_adv_loader = DataLoader(test_adv_dataset, batch_size=batch_size, shuffle=False)
    train_x_loader = DataLoader(train_x_dataset, batch_size=batch_size)
    train_adv_loader = DataLoader(train_adv_dataset, batch_size=batch_size)
    # for (x, x_y), (adv, adv_y) in zip(train_x_loader, train_adv_loader):
    #     if not torch.all(torch.eq(x_y, adv_y)):
    #         raise ValueError("data error")
    # for (x, x_y), (adv, adv_y) in zip(test_x_loader, test_adv_loader):
    #     if not torch.all(torch.eq(x_y, adv_y)):
    #         raise ValueError("data error")
    # classifier.eval()
    # denoiser = Denoiser(conf=conf, unet=unet, classifier=classifier)
    unet.load_state_dict(torch.load("predict_cond2.pth"))
    # denoiser.denoiser.eval()
    print(method)
    # 22
    opt = optim.Adam(unet.parameters(), lr=1e-4, amsgrad=False)

    def mse_loss(output, target):
        return F.mse_loss(output, target)

    best_test_loss = 1.0
    for i in range(conf["epochs"]):
        train_batch_count = 0
        test_batch_count = 0
        print(f"epcch: {i + 1}")
        unet.train()
        for (x, x_y), (adv, adv_y) in zip(train_x_loader, train_adv_loader):
            opt.zero_grad()
            x, adv = x.to(device), adv.to(device)
            # noise = torch.randn_like(adv)
            # x_noised = net(x, noise=noise)
            # adv_noised = net(adv, noise=noise)
            x_cond = get_cond(x)
            adv_cond = get_cond(adv)
            loss1 = mse_loss(x, adv_cond)
            inputs = torch.cat([x_cond, adv_cond], dim=0).to(device)
            predict_noise_x = unet(inputs)
            outputs = torch.cat([x, x], dim=0).to(device)
            # adv_noised = net(adv, predict_noise)
            # adv_noised.requires_grad_(True)
            loss = mse_loss(outputs, predict_noise_x)
            loss.backward()
            opt.step()
            if train_batch_count % 10 ==0:
                print(f"loss:{loss.item()}\nx_adv:{loss1.item()}")
            train_batch_count += 1
        # inputs = torch.cat([x, adv], dim=0).to(device)
        # labels = torch.cat([x, x], dim=0).to(device)
        # denoiser.start_training(inputs=inputs, lables=labels, method=method)
    # denoiser.loss_total = 0.0
        print(f"-------------start test---------------")
        unet.eval()
        test_loss = 0.0
        with torch.no_grad():
            for (x, x_y), (adv, adv_y) in zip(test_x_loader, test_adv_loader):
                    x, adv = x.to(device), adv.to(device)
                    x_cond = get_cond(x)
                    adv_cond = get_cond(adv)
                    loss2 = mse_loss(x, adv_cond)
                    # inputs = torch.cat([adv_noised, noise], dim=1).to(device)
                    predict_noise_x = unet(adv_cond)
                    # adv_noised = net(adv, predict_noise)
                    # adv_noised.requires_grad_(True)
                    loss = mse_loss(x, predict_noise_x)
                    # print(f"loss:{loss}")
                    test_loss += loss.item()
                    # print(f"test_loss:{test_loss}")
                    test_batch_count += 1
        test_loss = test_loss/test_batch_count
        print(f"test_loss: {test_loss:>7f}\nx-adv:{loss2.item()}")
        if best_test_loss > test_loss:
            best_test_loss = test_loss
            torch.save(unet.state_dict(), "predict_cond2.pth".format(method, DATA_TYPE))
            print("save model")
    # print(f"x_correct: {(denoiser.x_correct / test_size * 100)}, "
    #       f"denoised_correct: {(denoiser.denoised_correct / test_size * 100)}")
    # if denoiser.denoised_correct > best_denoise_correct:
    #     best_denoise_correct = denoiser.denoised_correct
    #     torch.save(denoiser.denoiser.state_dict(), "{}_{}.pth".format(method, DATA_TYPE))
    #     print("save model")
    # torch.save(denoiser.denoiser.state_dict(), "{}_{}_end.pth".format(method, DATA_TYPE))
    # print("save end model")
    # print(f"best:{best_denoise_correct}")
    # if not torch.all(torch.eq(x, adv)):
    #     print("1")
    # print(x.shape)
    # break
    # inputs = torch.cat([x_batch, x_adv_batch], dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='config/purification_sr3.json',
    #                     help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test', 'testn'], help='Run train or test',
                        default='train')
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
