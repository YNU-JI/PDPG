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
from unet2 import UNet
from val.utils import NpyDataset, get_model



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
        "epochs":110,
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
    MODEL_TYPE = 'alexnet'  # resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
    DATA_Model = 'alexnet'  # resnet50 vgg16 densenet VIT MaxVit lenet resnet18 alexnet
    DATA_TYPE = 'mnist'  # cifar10 imagenet50 mnist
    MODEL_PATH = '../../Model_Weight/'
    method = "HGD"  # DAE HGD
    classifier = get_model(MODEL_TYPE, DATA_TYPE, device, MODEL_PATH)
    # conf = json.dumps(conf)
    # unet = smp.Unet(in_channels=6, classes=6, decoder_channels=[128, 64, 32, 16, 8], encoder_weights=None, activation="tanh").to(device)
    unet = UNet(n_channels=1, n_classes=1).to(device)
    print(torch.initial_seed())
    # print(unet.encoder)
    # print(unet)
    # 打印输出形状
    # print(output.shape)
    test_x_data_root = f'/home/special/user/jijun/MNIST_data/test_npy'
    test_adv_data_root = f'/home/special/user/jijun/Adv_img/{DATA_TYPE}/test/{MODEL_TYPE}/cw'
    test_adv_dataset = NpyDataset(test_adv_data_root, DATA_TYPE)
    test_x_dataset = NpyDataset(test_x_data_root, DATA_TYPE)

    train_x_data_root = '/home/special/user/jijun/MNIST_data/train_npy'
    train_adv_data_root = f'/home/special/user/jijun/Adv_img/{DATA_TYPE}/train/{MODEL_TYPE}/pgd'
    train_x_dataset = NpyDataset(train_x_data_root, DATA_TYPE)
    train_adv_dataset = NpyDataset(train_adv_data_root, DATA_TYPE)

    # assert len(test_x_dataset) == len(test_adv_dataset)
    test_size = len(test_x_dataset)
    # print(len(train_x_dataset))
    # assert len(train_x_dataset) == len(train_adv_dataset)

    # print(len(x_dataset), len(adv_dataset))
    # 创建数据加载器
    batch_size = 2000
    test_x_loader = DataLoader(test_x_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    test_adv_loader = DataLoader(test_adv_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
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
    denoiser.denoiser.load_state_dict(torch.load("HGD_mnist.pth"))
    denoiser.denoiser.eval()
    print(method)

    best_denoise_correct = 0.0
    # for i in range(conf["epochs"]):
    #     print(f"epcch: {i+1}")
    #     denoiser.x_correct, denoiser.denoised_correct = 0.0, 0.0
    #     denoiser.loss_total = 0.0
    #     for (x,x_y), (adv,adv_y) in zip(train_x_loader, train_adv_loader):
    #         # min_val = torch.min(torch.flatten(x))  # 获取整个tensor中的最小值
    #         # max_val = torch.max(torch.flatten(x))  # 获取整个tensor中的最大值
    #         #
    #         # print(min_val)  # 打印整个tensor的最小值
    #         # print(max_val)  # 打印整个tensor的最大值
    #         # min_val = torch.min(torch.flatten(adv))  # 获取整个tensor中的最小值
    #         # max_val = torch.max(torch.flatten(adv))  # 获取整个tensor中的最大值
    #         #
    #         # print(min_val)  # 打印整个tensor的最小值
    #         # print(max_val)  # 打印整个tensor的最大值
    #         inputs = torch.cat([x, adv], dim=0).to(device)
    #         labels = torch.cat([x, x], dim=0).to(device)
    #         denoiser.start_training(inputs=inputs, lables=labels, method=method)
            # if denoiser.batch_num%100 == 0:
    denoiser.loss_total = 0.0
    denoiser.x_correct, denoiser.denoised_correct = 0.0, 0.0
    print(f"-------------start test---------------")
    for (x,x_y), (adv,adv_y) in zip(test_x_loader, test_adv_loader):
        y = x_y.to(device)
        inputs = adv.to(device)
        labels = x.to(device)
        denoiser.evaluate(inputs, labels, method, y)
    print(f"x_correct: {(denoiser.x_correct / test_size * 100)}, "
      f"denoised_correct: {(denoiser.denoised_correct / test_size * 100)}")
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