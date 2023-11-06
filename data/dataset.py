import random

import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox, add_noise)





IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        npy_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]
        sorted_npy_files = sorted(npy_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1].split(".")[0]))
        # print(f"{sorted_npy_files}")
    return sorted_npy_files

def pil_loader(path):
    return Image.open(path).convert('RGB')

def np_loader(path, data_type=None):
    data = np.load(path)
    data = np.squeeze(data)
    if data_type == 'mnist':
        data = np.expand_dims(data, axis=0)
    data = np.transpose(data, (1, 2, 0))
    return data

def np_loader_2(path):
    data = np.load(path)
    # data = np.squeeze(data)
    split_arr = np.split(data, 2, axis=0)
    img = np.squeeze(split_arr[0])
    img = np.transpose(img, (1, 2, 0))
    cond_image = np.squeeze(split_arr[1])
    cond_image = np.transpose(cond_image, (1, 2, 0))
    # data = torch.from_numpy(data)
    # data = np.transpose(data, (1, 2, 0))
    return img, cond_image


class InpaintDataset3(data.Dataset):
    def __init__(self, data_root, adv_root, img_opt, mode, mask_config={}, data_len=-1, image_size=[32, 32], loader=np_loader):
        # train_adv_root = '../Adv_img/cifar10/train/resnet50/pgd2'
        # test_root = '../Adv_img/cifar10/test/resnet50/pgd'  # fgsm bim pgd mim deepfool_0.3 cw
        imgs = make_dataset(data_root)
        if mode == "train":
            self.adv_imgs = make_dataset(adv_root)
        else:
            self.adv_imgs = make_dataset(adv_root)
        # print(self.adv_imgs)
        # print(img1)

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
            # print(self.imgs)
        else:
            self.imgs = imgs
            # print(self.imgs)
            # 1/0

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(p=1.0),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        # print("-----------------claen-------------")
        # print(path)
        # if self.adv_img is not None:
        # print("------------------adv--------------")
        adv_path = self.adv_imgs[index]
        # print(adv_path)
        # 1/0
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path))
        adv_img = self.tfs(self.loader(adv_path))
        if self.mode == "train":
            concatenated_image = torch.cat((img, adv_img), dim=0)
            randF = transforms.RandomHorizontalFlip(p=0.5)
            concatenated_image = randF(concatenated_image)
            img = concatenated_image[:3, :, :]
            adv_img = concatenated_image[3:, :, :]
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # print(eps)
        perturb_img = self.get_cond_img(adv_img=img, gt_img=img, im_index=im_index)
        cond_image = self.get_cond_img(adv_img=adv_img, gt_img=adv_img, im_index=im_index)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['adv_img'] = adv_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        current_seed = torch.initial_seed()
        # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False)
        # compressed_img_64 = compressed_img.squeeze(0)
        # upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
        #                               align_corners=False)
        # gt_img = upsampled_img.squeeze(0)
        gt_img = self.inv_normalize(gt_img)
        random.seed()
        random_int = random.randint(1, 1000)
        if self.mode == 'train':
            torch.manual_seed(random_int)
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            gt_img = torch.clamp(gt_img, min=0, max=1)
        perturb_img = self.normalize(gt_img)
        torch.manual_seed(current_seed)
        return perturb_img

    def get_cond_img(self,adv_img, gt_img, im_index):
        x_up = gt_img
        eps = 8 / 255
        # h_flip = transforms.RandomHorizontalFlip(p=1.0)
        # v_flip = transforms.RandomVerticalFlip(p=1.0)
        # gt_img = v_flip(h_flip(gt_img))
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            compressed_img_64 = tfg(compressed_img_64)
            upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
                                          align_corners=False)
            x_up = upsampled_img.squeeze(0)
            # x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img + noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        elif self.img_opt == 'other':
            current_seed = torch.initial_seed()
            eps = eps
            random.seed()
            random_int = current_seed
            # if self.mode == 'train':
            #     random_int = random.randint(1, 1000)
            for i in range(random_int, random_int + 50):
                torch.manual_seed(i)
                noise = torch.randn_like(gt_img)
                # # # noise = (noise * 2 - 1) * eps
                noise = torch.clamp(noise, -eps, eps)
                gt_img = gt_img + noise
                # print(gt_img[0])
            x_up = torch.clamp(gt_img, min=-1, max=1)
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1)
            x_up = tfg(x_up)
            torch.manual_seed(current_seed)
            # perturb_img = self.normalize(gt_img)
        elif self.img_opt == 'adv':
            noise = torch.randn_like(adv_img) * eps
            noise = torch.clamp(noise, -eps, eps)
            x_up = adv_img + noise
            if self.mode == 'train':
                random.seed()
                random_int = random.randint(1, 1000)
                adv_noise = adv_img - gt_img
                # if random_int < 50:
                #     x_up = gt_img
                # elif random_int < 900:
                x_up = adv_img + adv_noise * 1
            # tfg = transforms.GaussianBlur(kernel_size=3, sigma=1)
            # x_up = tfg(x_up)
            x_up = torch.clamp(x_up, min=-1, max=1)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps


class InpaintDataset3_wr(data.Dataset):
    def __init__(self, data_root, adv_root, img_opt, mode, mask_config={}, data_len=-1, image_size=[32, 32], loader=np_loader):
        # train_adv_root = '../Adv_img/cifar10/train/resnet50/pgd2'
        # test_root = '../Adv_img/cifar10/test/resnet50/pgd'  # fgsm bim pgd mim deepfool_0.3 cw
        imgs = make_dataset(data_root)
        if mode == "train":
            self.adv_imgs = make_dataset(adv_root)
        else:
            self.adv_imgs = make_dataset(adv_root)
        # print(self.adv_imgs)
        # print(img1)

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
            # print(self.imgs)
        else:
            self.imgs = imgs
            # print(self.imgs)
            # 1/0

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=1.0),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        # print("-----------------claen-------------")
        # print(path)
        # if self.adv_img is not None:
        # print("------------------adv--------------")
        adv_path = self.adv_imgs[index]
        # print(adv_path)
        # 1/0
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path))
        adv_img = self.tfs(self.loader(adv_path))
        if self.mode == "train":
            concatenated_image = torch.cat((img, adv_img), dim=0)
            randF = transforms.RandomHorizontalFlip(p=0.5)
            concatenated_image = randF(concatenated_image)
            img = concatenated_image[:3, :, :]
            adv_img = concatenated_image[3:, :, :]
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # img = (img * self.std.view(-1, 1, 1)) + self.mean.view(-1, 1, 1)
        # img = (img - 0.5) * 2
        adv_img = (adv_img * self.std.view(-1, 1, 1)) + self.mean.view(-1, 1, 1)
        adv_img = (adv_img - 0.5) * 2
        # min_val = torch.min(torch.flatten(img))  # 获取整个tensor中的最小值
        # max_val = torch.max(torch.flatten(img))  # 获取整个tensor中的最大值
        # print("得到-1,1的x_re\n")
        # print(min_val)  # 打印整个tensor的最小值
        # print(max_val)  # 打印整个tensor的最大值
        # print(eps)
        perturb_img = self.get_cond_img(adv_img=img, gt_img=img, im_index=im_index)
        cond_image = self.get_cond_img(adv_img=adv_img, gt_img=img, im_index=im_index)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        current_seed = torch.initial_seed()
        # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False)
        # compressed_img_64 = compressed_img.squeeze(0)
        # upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
        #                               align_corners=False)
        # gt_img = upsampled_img.squeeze(0)
        gt_img = self.inv_normalize(gt_img)
        random.seed()
        random_int = random.randint(1, 1000)
        if self.mode == 'train':
            torch.manual_seed(random_int)
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            gt_img = torch.clamp(gt_img, min=0, max=1)
        perturb_img = self.normalize(gt_img)
        torch.manual_seed(current_seed)
        return perturb_img

    def get_cond_img(self,adv_img, gt_img, im_index):
        x_up = gt_img
        eps = 8 / 255
        # h_flip = transforms.RandomHorizontalFlip(p=1.0)
        # v_flip = transforms.RandomVerticalFlip(p=1.0)
        # gt_img = v_flip(h_flip(gt_img))
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            compressed_img_64 = tfg(compressed_img_64)
            upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
                                          align_corners=False)
            x_up = upsampled_img.squeeze(0)
            # x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img + noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        elif self.img_opt == 'other':
            current_seed = torch.initial_seed()
            eps = eps
            random.seed()
            random_int = current_seed
            # if self.mode == 'train':
            #     random_int = random.randint(1, 1000)
            for i in range(random_int, random_int + 50):
                torch.manual_seed(i)
                noise = torch.randn_like(gt_img)
                # # # noise = (noise * 2 - 1) * eps
                noise = torch.clamp(noise, -eps, eps)
                gt_img = gt_img + noise
                # print(gt_img[0])
            x_up = torch.clamp(gt_img, min=-1, max=1)
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1)
            x_up = tfg(x_up)
            torch.manual_seed(current_seed)
            # perturb_img = self.normalize(gt_img)
        elif self.img_opt == 'adv':
            noise = torch.randn_like(adv_img) * eps
            noise = torch.clamp(noise, -eps, eps)
            x_up = adv_img + noise
            if self.mode == 'train':
                # random.seed()
                # random_int = random.randint(1, 1000)
                adv_noise = adv_img - gt_img
                # if random_int < 50:
                #     x_up = gt_img
                # elif random_int < 900:
                x_up = adv_img + adv_noise * 1
            # tfg = transforms.GaussianBlur(kernel_size=3, sigma=1)
            # x_up = tfg(x_up)
            x_up = torch.clamp(x_up, min=-1, max=1)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps


class InpaintDataset3_224(data.Dataset):
    def __init__(self, data_root, adv_root, img_opt, mode, mask_config={}, data_len=-1, image_size=[224, 224], loader=np_loader):
        # train_adv_root = '../Adv_img/cifar10/train/resnet50/pgd2'
        # test_root = '../Adv_img/cifar10/test/resnet50/pgd'  # fgsm bim pgd mim deepfool_0.3 cw
        imgs = make_dataset(data_root)
        if mode == "train":
            self.adv_imgs = make_dataset(adv_root)
        else:
            self.adv_imgs = make_dataset(adv_root)
        # print(self.adv_imgs)
        # print(img1)

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
            # print(self.imgs)
        else:
            self.imgs = imgs
            # print(self.imgs)
            # 1/0

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(p=1.0),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        # print("-----------------claen-------------")
        # print(path)
        # if self.adv_img is not None:
        # print("------------------adv--------------")
        adv_path = self.adv_imgs[index]
        # print(adv_path)
        # 1/0
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path))
        adv_img = self.tfs(self.loader(adv_path))
        if self.mode == "train":
            concatenated_image = torch.cat((img, adv_img), dim=0)
            randF = transforms.RandomHorizontalFlip(p=0.5)
            concatenated_image = randF(concatenated_image)
            img = concatenated_image[:3, :, :]
            adv_img = concatenated_image[3:, :, :]
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # print(eps)
        perturb_img = self.get_cond_img(adv_img=img, gt_img=img, im_index=im_index)
        cond_image = self.get_cond_img(adv_img=adv_img, gt_img=img, im_index=im_index)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['adv_img'] = adv_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        current_seed = torch.initial_seed()
        # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False)
        # compressed_img_64 = compressed_img.squeeze(0)
        # upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
        #                               align_corners=False)
        # gt_img = upsampled_img.squeeze(0)
        gt_img = self.inv_normalize(gt_img)
        random.seed()
        random_int = random.randint(1, 1000)
        if self.mode == 'train':
            torch.manual_seed(random_int)
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            gt_img = torch.clamp(gt_img, min=0, max=1)
        perturb_img = self.normalize(gt_img)
        torch.manual_seed(current_seed)
        return perturb_img

    def get_cond_img(self, adv_img, gt_img, im_index):
        x_up = gt_img
        eps = 8 / 255
        # h_flip = transforms.RandomHorizontalFlip(p=1.0)
        # v_flip = transforms.RandomVerticalFlip(p=1.0)
        # gt_img = v_flip(h_flip(gt_img))
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            compressed_img_64 = tfg(compressed_img_64)
            upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
                                          align_corners=False)
            x_up = upsampled_img.squeeze(0)
            # x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img + noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        elif self.img_opt == 'other':
            current_seed = torch.initial_seed()
            eps = eps
            random.seed()
            random_int = current_seed
            # if self.mode == 'train':
            #     random_int = random.randint(1, 1000)
            for i in range(random_int, random_int + 50):
                torch.manual_seed(i)
                noise = torch.randn_like(gt_img)
                # # # noise = (noise * 2 - 1) * eps
                noise = torch.clamp(noise, -eps, eps)
                gt_img = gt_img + noise
                # print(gt_img[0])
            x_up = torch.clamp(gt_img, min=-1, max=1)
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1)
            x_up = tfg(x_up)
            torch.manual_seed(current_seed)
            # perturb_img = self.normalize(gt_img)
        elif self.img_opt == 'adv':
            noise = torch.randn_like(adv_img) * eps
            noise = torch.clamp(noise, -eps, eps)
            x_up = adv_img + noise
            if self.mode == 'train':
                random.seed()
                random_int = random.randint(1, 1000)
                adv_noise = adv_img - gt_img
                # print(f"adv_noise:{adv_noise.sum().item()}")
                # if random_int < 50:
                #     x_up = gt_img
                # elif random_int < 900:
                x_up = adv_img + adv_noise * 1
            # tfg = transforms.GaussianBlur(kernel_size=3, sigma=1)
            # x_up = tfg(x_up)
            x_up = torch.clamp(x_up, min=-2.1179, max=2.6400)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps


class InpaintDataset3_28(data.Dataset):
    def __init__(self, data_root, adv_root, data_type, img_opt, mode, mask_config={}, data_len=-1, image_size=[28, 28], loader=np_loader):
        # train_adv_root = '../Adv_img/cifar10/train/resnet50/pgd2'
        # test_root = '../Adv_img/cifar10/test/resnet50/pgd'  # fgsm bim pgd mim deepfool_0.3 cw
        self.data_type = data_type
        imgs = make_dataset(data_root)
        if mode == "train":
            self.adv_imgs = make_dataset(adv_root)
        else:
            self.adv_imgs = make_dataset(adv_root)
        # print(self.adv_imgs)
        # print(img1)

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
            # print(self.imgs)
        else:
            self.imgs = imgs
            # print(self.imgs)
            # 1/0

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(p=1.0),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        # print("-----------------claen-------------")
        # print(path)
        # if self.adv_img is not None:
        # print("------------------adv--------------")
        adv_path = self.adv_imgs[index]
        # print(adv_path)
        # 1/0
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path, self.data_type))
        adv_img = self.tfs(self.loader(adv_path, self.data_type))
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # print(eps)
        perturb_img = self.get_cond_img(adv_img=img, gt_img=img, im_index=im_index)
        cond_image = self.get_cond_img(adv_img=adv_img, gt_img=img, im_index=im_index)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['cond_image'] = adv_img
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        current_seed = torch.initial_seed()
        # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False)
        # compressed_img_64 = compressed_img.squeeze(0)
        # upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
        #                               align_corners=False)
        # gt_img = upsampled_img.squeeze(0)
        gt_img = self.inv_normalize(gt_img)
        random.seed()
        random_int = random.randint(1, 1000)
        if self.mode == 'train':
            torch.manual_seed(random_int)
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            gt_img = torch.clamp(gt_img, min=0, max=1)
        perturb_img = self.normalize(gt_img)
        torch.manual_seed(current_seed)
        return perturb_img

    def get_cond_img(self, adv_img, gt_img, im_index):
        x_up = gt_img
        eps = 8 / 255
        # h_flip = transforms.RandomHorizontalFlip(p=1.0)
        # v_flip = transforms.RandomVerticalFlip(p=1.0)
        # gt_img = v_flip(h_flip(gt_img))
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            compressed_img_64 = tfg(compressed_img_64)
            upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
                                          align_corners=False)
            x_up = upsampled_img.squeeze(0)
            # x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img + noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        elif self.img_opt == 'other':
            current_seed = torch.initial_seed()
            eps = eps
            random.seed()
            random_int = current_seed
            # if self.mode == 'train':
            #     random_int = random.randint(1, 1000)
            for i in range(random_int, random_int + 50):
                torch.manual_seed(i)
                noise = torch.randn_like(gt_img)
                # # # noise = (noise * 2 - 1) * eps
                noise = torch.clamp(noise, -eps, eps)
                gt_img = gt_img + noise
                # print(gt_img[0])
            x_up = torch.clamp(gt_img, min=-1, max=1)
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1)
            x_up = tfg(x_up)
            torch.manual_seed(current_seed)
            # perturb_img = self.normalize(gt_img)
        elif self.img_opt == 'adv':
            noise = torch.randn_like(adv_img) * 0.3
            noise = torch.clamp(noise, -eps, eps)
            x_up = adv_img + noise
            if self.mode == 'train':
                random.seed()
                random_int = random.randint(1, 1000)
                adv_noise = adv_img - gt_img
                # if random_int < 50:
                #     x_up = gt_img
                # elif random_int < 900:
                x_up = adv_img + adv_noise * 2
            # tfg = transforms.GaussianBlur(kernel_size=3, sigma=1)
            # x_up = tfg(x_up)
            x_up = torch.clamp(x_up, min=0, max=1)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps

class InpaintDataset3_28_2(data.Dataset):
    def __init__(self, data_root, data_type, img_opt, mode, mask_config={}, data_len=-1, image_size=[28, 28], loader=np_loader):
        self.data_type = data_type
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([

                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(p=1.0),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        # self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path, self.data_type))
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # print(eps)
        perturb_img = self.get_perturb_img(gt_img=img, eps=0.3)
        cond_image = self.get_cond_img(gt_img=perturb_img, im_index=im_index, eps=eps)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        current_seed = torch.initial_seed()
        # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False)
        # compressed_img_64 = compressed_img.squeeze(0)
        # upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
        #                               align_corners=False)
        # gt_img = upsampled_img.squeeze(0)
        # gt_img = self.inv_normalize(gt_img)
        random.seed()
        random_int = random.randint(1, 1000)
        if self.mode == 'train':
            torch.manual_seed(random_int)
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            gt_img = torch.clamp(gt_img, min=0, max=1)
        perturb_img = gt_img
        torch.manual_seed(current_seed)
        return perturb_img

    def get_cond_img(self, gt_img, eps, im_index):
        x_up = gt_img
        eps = 0.3
        # h_flip = transforms.RandomHorizontalFlip(p=1.0)
        # v_flip = transforms.RandomVerticalFlip(p=1.0)
        # gt_img = v_flip(h_flip(gt_img))
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            compressed_img_64 = tfg(compressed_img_64)
            upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
                                          align_corners=False)
            x_up = upsampled_img.squeeze(0)
            # x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img  +  noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        elif self.img_opt == 'other':
            x_up = gt_img
            # current_seed = torch.initial_seed()
            # eps = 2 * eps
            # random.seed()
            # random_int = current_seed
            # for i in range(random_int, random_int+150):
            #     torch.manual_seed(i)
            #     noise = torch.randn_like(gt_img)
            #     # # # noise = (noise * 2 - 1) * eps
            #     noise = torch.clamp(noise, -eps, eps)
            #     gt_img = gt_img + noise
            #     # print(gt_img[0])
            # x_up = torch.clamp(gt_img, min=0, max=1)
            # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2)
            # x_up = tfg(x_up)
            # torch.manual_seed(current_seed)
            # perturb_img = self.normalize(gt_img)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps

class InpaintDataset3_224_2(data.Dataset):
    def __init__(self, data_root, img_opt, mode, mask_config={}, data_len=-1, image_size=[224, 224], loader=np_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([

                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=1.0),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean,
                         std=self.std)  # 归一化操作

        self.inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(self.mean, self.std)],
    std=[1/s for s in self.std]
)

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path))
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # print(eps)
        perturb_img = self.get_perturb_img(gt_img=img, eps=8 / 255)
        cond_image = self.get_cond_img(gt_img=perturb_img, im_index=im_index, eps=eps)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        current_seed = torch.initial_seed()
        # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False)
        # compressed_img_64 = compressed_img.squeeze(0)
        # upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
        #                               align_corners=False)
        # gt_img = upsampled_img.squeeze(0)
        gt_img = self.inv_normalize(gt_img)
        random.seed()
        random_int = random.randint(1, 1000)
        if self.mode == 'train':
            torch.manual_seed(random_int)
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            gt_img = torch.clamp(gt_img, min=0, max=1)
        perturb_img = self.normalize(gt_img)
        torch.manual_seed(current_seed)
        return perturb_img

    def get_cond_img(self, gt_img, eps, im_index):
        x_up = gt_img
        eps = 8/255
        # h_flip = transforms.RandomHorizontalFlip(p=1.0)
        # v_flip = transforms.RandomVerticalFlip(p=1.0)
        # gt_img = v_flip(h_flip(gt_img))
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(56, 56), mode='bicubic', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            # compressed_img_64 = tfg(compressed_img_64)
            upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(224, 224), mode='bicubic',
                                          align_corners=False)
            x_up = upsampled_img.squeeze(0)
            # x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img  +  noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        elif self.img_opt == 'other':
            # x_up = gt_img
            current_seed = torch.initial_seed()
            eps = 2 * eps
            random.seed()
            random_int = current_seed
            # if self.mode == 'train':
            #     random_int = random.randint(1, 1000)
            for i in range(random_int, random_int+150):
                torch.manual_seed(i)
                noise = torch.randn_like(gt_img)
                # # # noise = (noise * 2 - 1) * eps
                noise = torch.clamp(noise, -eps, eps)
                gt_img = gt_img + noise
                # print(gt_img[0])
            x_up = torch.clamp(gt_img, min=-2.1179, max=2.6400)
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=2)
            x_up = tfg(x_up)
            torch.manual_seed(current_seed)
            # perturb_img = self.normalize(gt_img)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, img_opt, mode, mask_config={}, data_len=-1, image_size=[32, 32], loader=np_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        if mode == "train":
            self.tfs = transforms.Compose([
                    # transforms.RandomCrop(image_size[0], padding=4),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=1.0),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([
                # transforms.RandomCrop(image_size[0], padding=4),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=1.0),
                # transforms.Resize((image_size[0], image_size[1])),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path))
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # print(eps)
        perturb_img = self.get_perturb_img(gt_img=img, eps=8/255)
        cond_image = self.get_cond_img(gt_img=perturb_img, im_index=im_index, eps=eps)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        perturb_img = gt_img
        if self.mode == 'train':
            noise = torch.randn_like(gt_img)
            # # # noise = (noise * 2 - 1) * eps
            noise = torch.clamp(noise, -eps, eps)
            x_noise = gt_img + noise
            x_noise = self.inv_normalize(x_noise)
            gt_img = torch.clamp(x_noise, min=0, max=1)
            perturb_img = self.normalize(gt_img)
        return perturb_img



    def get_cond_img(self, gt_img, eps, im_index):
        x_up = gt_img
        eps = 8 / 255
        if self.img_opt == 'compression':
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            # im_gray = gray_transform(gt_img)  # 转换为灰度图像
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            # im_gray = gray_transform(gt_img)  # 转换为灰度图像

            # x_up = torch.clamp(x_up, 0, 1)

            # gt_img = normalize(gt_img)
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
            compressed_img = compressed_img.squeeze(0)
            upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                          align_corners=False)
            gt_img = upsampled_img.squeeze(0)

            tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
            # im_gray = tfg(im_gray)
            x_up1 = torch.clamp(gt_img, 0, 1)
            x_up2 = torch.clamp(gt_img, -1, 0)
            compressed_img = F.interpolate(x_up1.unsqueeze(0), size=(8, 8), mode='bilinear', align_corners=False)
            compressed_img = compressed_img.squeeze(0)
            upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                          align_corners=False)
            x_up1 = upsampled_img.squeeze(0)

            compressed_img = F.interpolate(x_up2.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img = compressed_img.squeeze(0)
            upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                          align_corners=False)
            x_up2 = upsampled_img.squeeze(0)
            x_up2 = tfg(x_up2)
            # img_filtered = transforms.ToPILImage()(x_up2)
            # # 进行中值滤波
            # x_up2 = img_filtered.filter(ImageFilter.MedianFilter(size=3))
            # x_up2 = transforms.ToTensor()(x_up2)
            # x_up2 = 2 * x_up2 - 1   # 将0,1映射到-1,0
            x_up = x_up1 + x_up2
            x_up = self.inv_normalize(x_up)
            # min_val = torch.min(torch.flatten(x_up))  # 获取整个tensor中的最小值
            # max_val = torch.max(torch.flatten(x_up))  # 获取整个tensor中的最大值
            #
            # print(min_val)  # 打印整个tensor的最小值
            # print(max_val)  # 打印整个tensor的最大值
            x_up = torch.clamp(x_up, 0, 1)

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img + noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps



class InpaintDataset2(data.Dataset):
    def __init__(self, data_root, data_type, img_opt, mode, mask_config={}, data_len=-1, image_size=[32, 32], loader=np_loader):
        self.data_type = data_type
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomVerticalFlip(p=1.0),
                    # transforms.RandomRotation(degrees=15, fill=255),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([

                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0)
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path))
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # print(eps)
        perturb_img = self.get_perturb_img(gt_img=img, eps=8 / 255)
        cond_image = self.get_cond_img(gt_img=perturb_img, im_index=im_index, eps=eps)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        perturb_img = gt_img
        if self.mode == 'train':
            noise = torch.randn_like(gt_img)
            # # # noise = (noise * 2 - 1) * eps
            noise = torch.clamp(noise, -eps, eps)
            x_noise = gt_img + noise
            x_noise = self.inv_normalize(x_noise)
            gt_img = torch.clamp(x_noise, min=0, max=1)
            perturb_img = self.normalize(gt_img)
        return perturb_img

    def get_cond_img(self, gt_img, eps, im_index):
        x_up = gt_img
        eps = 8/255
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            compressed_img_64 = tfg(compressed_img_64)
            x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img  +  noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps


class InpaintDataset33(data.Dataset):
    def __init__(self, data_root, img_opt, mode, mask_config={}, data_len=-1, image_size=[32, 32], loader=np_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([

                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=1.0),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path))
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # print(eps)
        perturb_img = self.get_perturb_img(gt_img=img, eps=8 / 255)
        cond_image = self.get_cond_img(gt_img=perturb_img, im_index=im_index, eps=eps)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        current_seed = torch.initial_seed()
        # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False)
        # compressed_img_64 = compressed_img.squeeze(0)
        # upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
        #                               align_corners=False)
        # gt_img = upsampled_img.squeeze(0)
        gt_img = self.inv_normalize(gt_img)
        random.seed()
        random_int = random.randint(1, 1000)
        if self.mode == 'train':
            torch.manual_seed(random_int)
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            gt_img = torch.clamp(gt_img, min=0, max=1)
        perturb_img = self.normalize(gt_img)
        torch.manual_seed(current_seed)
        return perturb_img

    def get_cond_img(self, gt_img, eps, im_index):
        x_up = gt_img
        eps = 8/255
        # h_flip = transforms.RandomHorizontalFlip(p=1.0)
        # v_flip = transforms.RandomVerticalFlip(p=1.0)
        # gt_img = v_flip(h_flip(gt_img))
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            compressed_img_64 = tfg(compressed_img_64)
            upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
                                          align_corners=False)
            x_up = upsampled_img.squeeze(0)
            # x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img  +  noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        elif self.img_opt == 'other':
            x_up = gt_img
            # current_seed = torch.initial_seed()
            # eps = 0.1
            # random.seed()
            # random_int = current_seed
            # if self.mode == 'train':
            #     random_int = random.randint(1, 1000)
            # for i in range(random_int, random_int+16):
            #     torch.manual_seed(i)
            #     noise = torch.randn_like(gt_img)
            #     # # # noise = (noise * 2 - 1) * eps
            #     noise = torch.clamp(noise, -eps, eps)
            #     gt_img = gt_img + noise
            #     # print(gt_img[0])
            # x_up = torch.clamp(gt_img, min=-1, max=1)
            # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2)
            # x_up = tfg(x_up)
            # torch.manual_seed(current_seed)
            # perturb_img = self.normalize(gt_img)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps






class InpaintDataset3_2(data.Dataset):
    def __init__(self, data_root, img_opt, mode, mask_config={}, data_len=-1, image_size=[32, 32], loader=np_loader_2):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([

                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(p=1.0),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img, cond_image = self.loader(path)
        img = self.tfs(img)
        # print(img.shape)
        cond_image = self.tfs(cond_image)
        # print(f"img:{img.max()}, {img.min()}")
        mask1, mask2, eps = self.get_mask()
        # print(eps)
        # perturb_img = self.get_perturb_img(gt_img=img, eps=8 / 255)
        # cond_image = self.get_cond_img(gt_img=perturb_img, im_index=im_index, eps=eps)
        mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = cond_image
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask2
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        current_seed = torch.initial_seed()
        # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False)
        # compressed_img_64 = compressed_img.squeeze(0)
        # upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
        #                               align_corners=False)
        # gt_img = upsampled_img.squeeze(0)
        gt_img = self.inv_normalize(gt_img)
        random.seed()
        random_int = random.randint(1, 1000)
        if self.mode == 'train':
            torch.manual_seed(random_int)
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            gt_img = torch.clamp(gt_img, min=0, max=1)
        perturb_img = self.normalize(gt_img)
        torch.manual_seed(current_seed)
        return perturb_img

    def get_cond_img(self, gt_img, eps, im_index):
        x_up = gt_img
        eps = 8/255
        # h_flip = transforms.RandomHorizontalFlip(p=1.0)
        # v_flip = transforms.RandomVerticalFlip(p=1.0)
        # gt_img = v_flip(h_flip(gt_img))
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            compressed_img_64 = tfg(compressed_img_64)
            upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
                                          align_corners=False)
            x_up = upsampled_img.squeeze(0)
            # x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img  +  noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        elif self.img_opt == 'other':
            current_seed = torch.initial_seed()
            eps = 2 * eps
            random.seed()
            random_int = current_seed
            for i in range(random_int, random_int+150):
                torch.manual_seed(i)
                noise = torch.randn_like(gt_img)
                # # # noise = (noise * 2 - 1) * eps
                noise = torch.clamp(noise, -eps, eps)
                gt_img = gt_img + noise
                # print(gt_img[0])
            x_up = torch.clamp(gt_img, min=-1, max=1)
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=2)
            x_up = tfg(x_up)
            torch.manual_seed(current_seed)
            # perturb_img = self.normalize(gt_img)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        elif self.mask_mode == 'noise':
            mask1, mask2, eps = add_noise(self.image_size)
            # self.image_size.append(1)
            # mask.reshape(self.image_size)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask1).permute(2,0,1),torch.from_numpy(mask2).permute(2,0,1), eps

class InpaintDataset4(data.Dataset):
    def __init__(self, data_root, img_opt, mode, mask_config={}, data_len=-1, image_size=[32, 32], loader=np_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        if mode == "train":
            self.tfs = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.RandomCrop(image_size[0], padding=2, padding_mode='edge'),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    # transforms.Resize((image_size[0], image_size[1])),

                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
            ])
        else:
            # random_seed = 0
            # random.seed(random_seed)
            # np.random.seed(random_seed)
            # torch.manual_seed(random_seed)
            # torch.cuda.manual_seed_all(random_seed)
            self.tfs = transforms.Compose([

                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=1.0),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.Resize((image_size[0], image_size[1])),

                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.img_opt = img_opt
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        self.inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        filename = os.path.basename(path)
        filebase, fileext = os.path.splitext(filename)
        label, im_index = filebase.split('_')[0], int(filebase.split('_')[1])
        label = torch.tensor(int(label))
        img = self.tfs(self.loader(path))
        # print(f"img:{img.max()}, {img.min()}")
        # print(eps)
        perturb_img = self.get_perturb_img(gt_img=img, eps=8 / 255)
        mask = self.get_mask()
        cond_image = perturb_img * (1. - mask) + mask * torch.randn_like(perturb_img)
        mask_img = perturb_img * (1. - mask) + mask
        # cond_image = self.get_cond_img(gt_img=perturb_img, im_index=im_index, eps=eps)
        # mask_img = img * (1. - mask2) + mask2
        ret['label'] = label
        ret['gt_image'] = img
        ret['perturb_image'] = perturb_img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # else:
        #     # 添加黑白噪声
        #     # noise = torch.randn_like(img[:1, :, :])
        #     # noise = noise.repeat(3, 1, 1)
        #     # 添加噪音
        #     noise = torch.randn_like(img)
        #     cond_image = img * (1. - mask1) +  noise * eps
        #     # print(f"{cond_image.max()}, {cond_image.min()}")
        #     # 10%是干净样本，50%是噪声样本
        #     if torch.rand(1) <= 0.5:
        #         # 将像素值限制在[0,1]范围内
        #         cond_image = torch.clamp(cond_image, min=-1, max=1)
        #     else:
        #         cond_image = img
        #     mask_img = img * (1. - mask2) + mask2
        #     ret['label'] = label
        #     ret['gt_image'] = img
        #     ret['cond_image'] = cond_image
        #     ret['mask_image'] = mask_img
        #     ret['mask'] = mask2
        #     ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_perturb_img(self, gt_img, eps):
        current_seed = torch.initial_seed()
        # tfg = transforms.GaussianBlur(kernel_size=3, sigma=2.0)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False)
        # compressed_img_64 = compressed_img.squeeze(0)
        # upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
        #                               align_corners=False)
        # gt_img = upsampled_img.squeeze(0)
        gt_img = self.inv_normalize(gt_img)
        random.seed()
        random_int = random.randint(1, 1000)
        if self.mode == 'train':
            torch.manual_seed(random_int)
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            gt_img = torch.clamp(gt_img, min=0, max=1)
        perturb_img = self.normalize(gt_img)
        torch.manual_seed(current_seed)
        return perturb_img

    def get_cond_img(self, gt_img, eps, im_index):
        x_up = gt_img
        eps = 8/255
        h_flip = transforms.RandomHorizontalFlip(p=1.0)
        v_flip = transforms.RandomVerticalFlip(p=1.0)
        gt_img = v_flip(h_flip(gt_img))
        # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        # inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        if self.img_opt == 'compression':
            tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
            # gray_transform = transforms.Grayscale()  # 定义灰度转换器
            compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
            compressed_img_64 = compressed_img.squeeze(0)
            compressed_img_64 = tfg(compressed_img_64)
            upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(32, 32), mode='bicubic',
                                          align_corners=False)
            x_up = upsampled_img.squeeze(0)
            # x_up = TF.pad(compressed_img_64, [8, 8, 8, 8], 0)
            # x_up = inv_normalize(x_up)
            # print(x_up.shape)
            # x_up = x_up * 0.9 + im_gray * 0.1

        elif self.img_opt == 'mix':
            if im_index % 10 < 3:
                noise = torch.randn_like(gt_img)
                x_up = gt_img  +  noise * eps
                x_up = torch.clamp(x_up, min=-1, max=1)
            else:
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False)
                compressed_img = compressed_img.squeeze(0)
                upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
        return x_up

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        # 添加黑白噪声
        noise = torch.randn_like(img[:1, :, :])
        noise = noise.repeat(3, 1, 1)
        cond_image = img*(1. - mask) + mask*noise
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


