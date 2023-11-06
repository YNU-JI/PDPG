
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


def wavelet_denoise(img, wavelet='db1', level=1, threshold_type='hard', threshold=0.1):
    # 将图像转换为张量
    img_tensor = torch.from_numpy(np.array(img)).float()

    # 进行小波变换
    coeffs = pywt.wavedec2(img_tensor, wavelet, level=level)

    # 对低频系数矩阵进行阈值处理
    coeffs = list(coeffs)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, threshold_type)

    # 进行逆小波变换
    img_denoised = pywt.waverec2(coeffs, wavelet)

    # 将张量转换为图像
    img_denoised = np.clip(img_denoised, 0, 255)
    img_denoised = img_denoised.astype(np.uint8)
    return img_denoised


class ImageProcessor:

    def __init__(self, img_opt='compression', mode='train'):
        self.img_opt = img_opt
        self.mode = mode

    def get_cond_img(self, gt_img, gt2_img, eps, im_index):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化操作
        inv_normalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # 反归一化操作
        gt2_img = gt_img
        x_up = gt_img
        randF = transforms.RandomHorizontalFlip(1.0)
        concatenated_image = torch.cat((gt_img, normalize(gt_img)), dim=0)
        rand_img = randF(concatenated_image)
        rand_img1 = rand_img[3:, :, :]
        rand_img = rand_img[:3, :, :]
        img_filtered = transforms.ToPILImage()(gt_img)
        # # 进行中值滤波
        compressed_img = img_filtered.filter(ImageFilter.MedianFilter(size=3))
        img_filtered = transforms.ToTensor()(compressed_img)

        eps = 16 / 255

        if self.img_opt == 'compression':
            if self.mode == "train":
                noise_img = gt_img
                for i in range(100):
                    print(i)
                    torch.manual_seed(i)
                    noise = torch.randn_like(noise_img)
                    # # # noise = (noise * 2 - 1) * eps
                    noise = torch.clamp(noise, -eps, eps)
                    noise_img = noise_img + noise
                    # print(noise_img[0])
                # x_noise = self.inv_normalize(x_noise)
                noise_img = torch.clamp(noise_img, min=0, max=1)
                tfg = transforms.GaussianBlur(kernel_size=3, sigma=2)
                col = transforms.ColorJitter(0.1,0.1,0.1,0.1)
                adv_img = col(gt_img)
                # adv_img = gt_img.clone()
                # for c in range(3):
                #     temp = (adv_img[c].cpu() * 255).numpy()
                #     img_dct = cv2.dct(temp)  # 进行离散余弦变换
                #
                #     for i in range(32):
                #         for j in range(32):
                #             if i + j > 32:
                #                 img_dct[i, j] = 0
                #     img_recor2 = cv2.idct(img_dct)
                #     adv_img[c] = torch.tensor(img_recor2) / 255
                # adv_img = tfg(adv_img)
                # adv_img = transforms.ToPILImage()(adv_img)
                # # # 进行中值滤波
                # adv_img = adv_img.filter(ImageFilter.MedianFilter(size=5))
                # adv_img = transforms.ToTensor()(adv_img)
                resize = transforms.Resize(36)
                img_36 = resize(gt_img)
                gamma_img = TF.adjust_gamma(gt_img, gamma=1.5)
                gamma_img = tfg(gamma_img)

                gt_img = normalize(gt_img)
                x_up = gt_img
                noise = torch.randn_like(gt_img) * eps
                # noise = (noise * 2 - 1) * eps
                noise = torch.clamp(noise, -eps, eps)
                x_noise = gt_img + noise
                noise_img = torch.clamp(x_noise, min=-1, max=1)
                noise_img = inv_normalize(noise_img)
                gray_transform = transforms.Grayscale()  # 定义灰度转换器
                im_gray = gray_transform(gt_img)  # 转换为灰度图像
                print(im_gray.shape)
                roate = transforms.RandomRotation(degrees=15, fill=255)

                # transforms.TrivialAugmentWide
                roate_img = roate(gt_img)
                # x_up = torch.clamp(x_up, 0, 1)

                # gt_img = normalize(gt_img)

                # im_gray = tfg(im_gray)

                # x_up = x_up * 0.3 + im_gray * 0.7
                gt_img_1_0 = torch.clamp(gt_img, -1, 0)
                # compressed_img = F.interpolate(x_up1.unsqueeze(0), size=(12, 12), mode='bicubic', align_corners=False)
                # compressed_img = compressed_img.squeeze(0)
                # upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bicubic',
                #                               align_corners=False)
                # x_up1 = upsampled_img.squeeze(0)
                gt_img_0_1 = torch.clamp(gt_img, 0, 1)
                # gt_img =gt_img_1_0
                gt_img = inv_normalize(gt_img)
                compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
                compressed_img_64 = compressed_img.squeeze(0)
                compressed_img_64 = tfg(compressed_img_64)
                img_pad = TF.pad(compressed_img_64,  [8,8,8,8], padding_mode='edge')


                x_up1 = compressed_img
                # img_filtered = transforms.ToPILImage()(compressed_img)
                # # # 进行中值滤波
                # compressed_img = img_filtered.filter(ImageFilter.MedianFilter(size=3))
                # compressed_img = transforms.ToTensor()(compressed_img)
                upsampled_img = F.interpolate(compressed_img_64.unsqueeze(0), size=(64, 64), mode='bicubic',
                                              align_corners=False)
                upsampled_img = upsampled_img.squeeze(0)
                upsampled_img = tfg(upsampled_img)

                upsampled_img = F.interpolate(upsampled_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                                              align_corners=False)
                x_up = upsampled_img.squeeze(0)
                # x_up = torch.clamp(normalize(x_up), 0, 1) + torch.clamp(normalize(x_up), -1, 0)

                x_up = inv_normalize(x_up)
                # x_up = tfg(x_up)

                # x_up2 = tfg(x_up2)
                # x_up = inv_normalize( x_up2)
                # x_up = x_up2
                # print(x_up.shape)
                # print(img_pad.shape)
                switch = random.randint(1, 4)
                print(switch)
                # compressed_img = F.interpolate(gt_img.unsqueeze(0), size=(16, 16), mode='bilinear', align_corners=False)
                # compressed_img = compressed_img.squeeze(0)
                # upsampled_img = F.interpolate(compressed_img.unsqueeze(0), size=(32, 32), mode='bilinear',
                #                               align_corners=False)
                # gt_img = upsampled_img.squeeze(0)
                #
                # x_up1 = torch.clamp(gt_img, 0, 1)
                # x_up2 = torch.clamp(gt_img, -1, 0)
                # # x_up2 = tfg(x_up2)
                #
                # x_up = x_up2 + x_up1
                # x_up = inv_normalize(x_up)
                # x_up = im_gray
                # tfg = transforms.GaussianBlur(kernel_size=3, sigma=1.2)
                # x_up = tfg(x_up)
        # denormalize the tensor
        # gt_img = gt_img * 0.5 + 0.5

        # convert the tensor to a PIL image
        gt_img_1_0 = transforms.ToPILImage()(gt_img_1_0)
        gt2_img = transforms.ToPILImage()(gt2_img)
        img_filtered = transforms.ToPILImage()(img_filtered)
        gamma_img = transforms.ToPILImage()(gamma_img)
        x_noise = transforms.ToPILImage()(noise_img)
        img_36 = transforms.ToPILImage()(img_36)
        gt_img = transforms.ToPILImage()(gt_img)
        adv_img = transforms.ToPILImage()(adv_img)
        noise_img = transforms.ToPILImage()(noise_img)
        roate_img = transforms.ToPILImage()(roate_img)
        rand_img = transforms.ToPILImage()(rand_img)
        rand_img1 = transforms.ToPILImage()(rand_img1)
        img_pad = transforms.ToPILImage()(img_pad)
        gt_img_0_1 = transforms.ToPILImage()(gt_img_0_1)
        compressed_img_64 = transforms.ToPILImage()(compressed_img_64)
        gt_img_pil = transforms.ToPILImage()(x_up)

        # save the image to disk
        img_pad.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_pad.png")
        gt2_img.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_gt2.png")
        img_filtered.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_filter.png")
        gamma_img.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_gamma.png")
        x_noise.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_x_noise.png")
        img_36.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_36.png")
        adv_img.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_dct.png")
        gt_img.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_gt.png")
        noise_img.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_noise.png")
        roate_img.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_roate.png")
        rand_img.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_rand.png")
        rand_img1.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_rand1.png")
        gt_img_1_0.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_1_0.png")
        gt_img_0_1.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_0_1.png")
        compressed_img_64.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_64.png")
        gt_img_pil.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/{im_index}_processed.png")

        return x_up

    # def median_filter(self, image, kernel_size=3, im_index=0):
    #     # image: 4D tensor [batch_size, channels, height, width]
    #     # kernel_size: size of the median filter kernel
    #
    #     # pad input image with zeros to preserve size
    #     padding = (kernel_size - 1) // 2
    #     padded_image = F.pad(image, (padding, padding, padding, padding), mode='constant', value=0)
    #
    #     # perform median filtering
    #     filtered_image = median_filter(image, kernel_size)
    #     gt_img_pil = transforms.ToPILImage()(filtered_image)
    #
    #     # save the image to disk
    #     gt_img_pil.save(f"/home/special/user/jijun/Adv_img/cifar10/test/resnet50/cleanimage_{im_index}_processed.png")
    #     return filtered_image

# load an example image
image_path = "/home/special/user/jijun/I2I/config/Cond_0_0_10.png"
image2_path = "/home/special/user/jijun/I2I/config/GT_0_0_10.png"
gt_img_pil = Image.open(image_path)
gt2_img_pil = Image.open(image2_path)
"""
/home/special/user/jijun/Adv_img/cifar10/test/resnet50/clean/clean_dir/1/6431.png
/home/special/user/jijun/Adv_img/cifar10/test/pgd_test/1/0_6431.png
/home/special/user/jijun/Adv_img/cifar10/test/resnet50/pgd_test/resnet50_out/2/3_2735.png
/home/special/user/jijun/Adv_img/cifar10/test/resnet50/pgd_test/resnet50_out/0/0_1077.png
"""
# convert the PIL image to a tensor
gt_img_tensor = transforms.ToTensor()(gt_img_pil)
gt2_img_tensor = transforms.ToTensor()(gt2_img_pil)
# print(gt_img_tensor)
# create an instance of ImageProcessor
processor = ImageProcessor()

# process the image and save the result
x_up = processor.get_cond_img(gt_img_tensor, gt2_img_tensor, eps=8 / 255, im_index=0)
# x_up = processor.median_filter(gt_img_tensor)
noise = None
noise = default(noise, lambda: torch.randn_like(gt_img_tensor))
noise2 = default(noise, lambda: torch.randn_like(noise))
# print(noise2 == noise)
# # 对图片进行中值滤波
# filtered_img = torch.nn.functional.median_filter2d(gt_img_tensor, kernel_size=3)
#
# # 将Tensor转换回PIL Image
# filtered_img_pil = transforms.ToPILImage()(filtered_img)
#
# # 保存滤波后的图片
# filtered_img_pil.save('/home/special/user/jijun/Adv_img/cifar10/test/resnet50/filtered_image.jpg')




# from PIL import Image, ImageFilter
# # # /home/special/user/jijun/Adv_img/cifar10/test/resnet50/pgd_test/resnet50_out/0/0_1077.png
# # # 读取图片
# img = Image.open('/home/special/user/jijun/Adv_img/cifar10/test/resnet50/clean/clean_dir/0/1077.png')
# # # /home/special/user/jijun/Adv_img/cifar10/test/resnet50/clean/clean_dir/0/1077.png
# #
# img_filtered = transforms.ToTensor()(img)
# eps = 8 / 255
# noise = torch.randn_like(img_filtered)
# noise = (noise * 2 - 1) * eps
# noise = torch.clamp(noise, -eps, eps)
# x_noise = img_filtered + noise
# img_filtered = torch.clamp(x_noise, min=-1, max=1)
# img_filtered = transforms.ToPILImage()(img_filtered)
# # 进行中值滤波
# img_filtered = img_filtered.filter(ImageFilter.MedianFilter(size=3))
# # 保存结果
# img_filtered.save('/home/special/user/jijun/Adv_img/cifar10/test/resnet50/filtered_image_noise.jpg')


# 读取图像并转换为灰度图
# im_gray = Image.open('/home/special/user/jijun/Adv_img/cifar10/test/resnet50/pgd_test/resnet50_out/1/0_1182.png').convert('L')
# im_gray.save('/home/special/user/jijun/Adv_img/cifar10/test/resnet50/gray.png')
# 将 PIL 图像转换为 numpy 数组

# 进行中值滤波
# blurred_img = im_gray.filter(ImageFilter.MedianFilter(size=3))
# x_up = transforms.ToTensor()(x_up)

# # 设置双边滤波参数
# kernel_size = 3
# sigma_color = 10
# sigma_space = 10
# blurred_tensor = TF.gaussian_blur(img_tensor, kernel_size,  sigma_space)

# 将 Tensor 转换为 PIL 图像并保存
# blurred_img = TF.to_pil_image(blurred_tensor)
# 保存图像
# blurred_img.save('/home/special/user/jijun/Adv_img/cifar10/test/resnet50/denoised.png')
# model = torchvision.models.wide_resnet50_2(pretrained=True)
# model.eval()

# pred = [[-1] * 11 for _ in range(10000)]
# print(pred[8888][0])



# 加载 .npy 文件
data_np = np.load('/home/special/user/jijun/I2I/experiments/pgd_cated/results/test/0/0_0_1010.npy')

split_arr = np.split(data_np, 2, axis=0)
# 将 NumPy 数组转换为张量
tensor1 = torch.from_numpy(split_arr[0])
tensor2 = torch.from_numpy(split_arr[1])
from torchvision.utils import save_image
tensor1 = np.squeeze(split_arr[0])
tensor1 = torch.from_numpy(tensor1)
# 拆分张量
# tensor1, tensor2 = torch.split(data_tensor, 1, dim=0)
print(tensor1.shape)
# 保存拆分后的张量为图像
torchvision.utils.save_image(tensor1, 'tensor1.png')
torchvision.utils.save_image(tensor2[0], 'tensor2.png')
# 保存拆分后的图像
# torchvision.utils.save_image(tensor1,'image1.png')
# torchvision.utils.save_image(tensor2,'image2.png')