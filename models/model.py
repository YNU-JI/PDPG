import random

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import Conv2d
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn.functional as F
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
from core import util
from val.utils import get_model
# from val.network import *
from .unet import UNet as PU
import os
import sys
sys.path.insert(0, './val')
current_path = os.getcwd()
print("Current working directory:", current_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
predict_cond = PU(n_channels=3, n_classes=3).to(device)
predict_cond.load_state_dict(torch.load("compare_exp/predict_cond2.pth"))
predict_cond.eval()

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, only_out, data_type, classifier,classifier_path, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        self.classifier_path = classifier_path
        self.classifier = classifier
        self.data_type = data_type
        self.cat_img = None
        self.pred = None
        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        self.size = 0
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None

        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        # optG2 = torch.optim.Adadelta(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        # self.scheduler = lr_scheduler.StepLR(self.optG, step_size=30, gamma=0.5)
        # self.optimizers.append(optG2)
        # self.optimizers.append(optG1)
        # self.schedulers.append(self.scheduler)
        self.resume_training()

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        self.only_out = only_out
        self.n_pred = [[-1] * 11 for _ in range(10000)]
        self.flag = 0

    def get_cond_img(self, gt_img):
        current_seed = torch.initial_seed()
        eps = 8 / 255
        if self.phase == 'train':
            torch.manual_seed(self.epoch % 10)  # 设置随机种子
            # print(f"seed: {torch.initial_seed()}")
            noise = torch.randn_like(gt_img)
            noise = torch.clamp(noise, -eps, eps)
            x_noise = gt_img + noise
            gt_img = torch.clamp(x_noise, min=-1, max=1)
        eps = 2 * eps
        for i in range(150):
            torch.manual_seed(i)
            noise = torch.randn_like(gt_img)
            # # # noise = (noise * 2 - 1) * eps
            noise = torch.clamp(noise, -eps, eps)
            gt_img = gt_img + noise
            # print(gt_img[0])
        x_up = torch.clamp(gt_img, min=-1, max=1)
        # tfg = transforms.GaussianBlur(kernel_size=5, sigma=0.7)
        # gt_img = tfg(gt_img)
        # compressed_img = F.interpolate(gt_img, size=(16, 16), mode='bilinear', align_corners=False)
        # upsampled_img = F.interpolate(compressed_img, size=(32, 32), mode='bicubic', align_corners=False)
        # x_up = upsampled_img
        torch.manual_seed(current_seed)
        return x_up

    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.perturb_image = self.set_device(data.get('perturb_image'))
        self.adv_img = self.set_device(data.get('adv_img'))
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        # self.cond_image = self.set_device(self.get_cond_img(self.gt_image))
        self.label = self.set_device(data.get('label'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])

    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu() + 1) / 2,
            'cond_image': (self.cond_image.detach()[:].float().cpu() + 1) / 2,
        }
        # if self.task in ['inpainting','uncropping']:
        #     dict.update({
        #         'mask': self.mask.detach()[:].float().cpu(),
        #         'mask_image': (self.mask_image+1)/2,
        #     })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu() + 1) / 2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        if self.only_out == 'count':
            for idx in range(self.batch_size):
                parts = self.path[idx][:-4].split("_")
                label, img_id = int(parts[0]), int(parts[-1])
                pred = int(self.pred[idx].detach().cpu())
                self.n_pred[img_id][10] = label
                self.n_pred[img_id][pred] += 1
        elif self.only_out == 'out':
            for idx in range(self.batch_size):
                if int(self.path[idx].split('_')[0]) != int(self.pred[idx].detach().cpu()) or 1==1:
                    # pass
                    # print(self.path[idx])
                    # ret_path.append('GT_{}_{}.png'.format(self.pred[idx].detach().cpu(), self.path[idx][:-4]))
                    # ret_result.append(self.gt_image[idx].detach().float().cpu())
                    #
                    # ret_path.append('Cond_{}_{}.png'.format(self.pred[idx].detach().cpu(), self.path[idx][:-4]))
                    # ret_result.append(self.cond_image[idx].detach().float().cpu())

                    # ret_path.append('adv_{}_{}.png'.format(self.pred[idx].detach().cpu(), self.path[idx][:-4]))
                    # ret_result.append(self.adv_img[idx].detach().float().cpu())

                    ret_path.append('Out_{}_{}.png'.format(self.pred[idx].detach().cpu(), self.path[idx][:-4]))
                    ret_result.append(self.output[idx].detach().float().cpu())

                    ret_path.append('{}_Process_{}.png'.format(self.pred[idx].detach().cpu(), self.path[idx][:-4]))
                    ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            if self.task in ['inpainting', 'uncropping']:
                pass
                # ret_path.extend(['Mask_{}.png'.format(name) for name in self.path[:-4]])
                # ret_result.extend(self.mask_image)
        else:
            for idx in range(self.batch_size):
                ret_path.append('{}'.format(self.path[idx]))
                ret_result.append(self.cat_img[idx::self.batch_size].detach().float().cpu())
        # if self.task in ['inpainting','uncropping']:
        #     ret_path.extend(['Mask_{}'.format(name) for name in self.path])
        #     ret_result.extend(self.mask_image)
        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
            # print(str(self.optG), self.optimizers.__len__())
        # elif self.epoch >= 400 and self.optimizers.__len__() > 0:
        #     self.optG = self.optimizers.pop()
        #     print(str(self.optG))
        # print(str(self.optG.param_groups[0]['lr']))
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, self.perturb_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self):
        self.logger.info(torch.initial_seed())
        self.logger.info(np.random.seed())
        self.logger.info(random.seed())
        """
                    import classifier
                    """
        DATA_TYPE = self.data_type
        MODEL_TYPE = self.classifier
        MODEL_PATH = self.classifier_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = get_model(MODEL_TYPE, DATA_TYPE, device, MODEL_PATH)
        print("classifier is inited!")
        # val: train_noise_schedule->test_noise_schedule
        self.netG.set_new_noise_schedule(phase="test")
        self.netG.eval()
        self.size = 0
        self.correct = 0
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                self.size += len(self.gt_image)
                if self.opt['distributed']:
                    if self.task in ['inpainting', 'uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image,
                                                                                 sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image,
                                                                                 sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting', 'uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                                                                                     y_0=self.gt_image, mask=self.mask,
                                                                                     sample_num=self.sample_num)
                        min_val = torch.min(torch.flatten(self.output))  # 获取整个tensor中的最小值
                        max_val = torch.max(torch.flatten(self.output))  # 获取整个tensor中的最大值

                        print(min_val)  # 打印整个tensor的最小值
                        print(max_val)  # 打印整个tensor的最大值
                        # tsf = transforms.RandomVerticalFlip(p=1.0)
                        # self.output = tsf(self.output)
                        self.evaluate(X=self.output, model=classifier, y=self.label, device=device)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                if self.epoch % 1 == 0:
                    for key, value in self.get_current_visuals(phase='val').items():
                        self.writer.add_images(key, value)
                    if self.only_out == 'cat':
                        self.writer.save_npy(self.save_current_results())
                    else:
                        self.writer.save_images(self.save_current_results())
            self.correct /= self.size
            correct = 100 * self.correct
            self.logger.info(f"val:\n Accuracy:{correct} \n")
            print(f"val:\n Accuracy:{correct} \n")
        # val_end: test_noise_schedule->train_noise_schedule
        self.netG.set_new_noise_schedule(phase=self.phase)
        return self.val_metrics.result()

    def test(self):
        print(self.data_type, self.classifier)
        self.logger.info(torch.initial_seed())
        self.logger.info(np.random.seed())
        self.logger.info(random.seed())
        """
            import classifier
            """
        DATA_TYPE = self.data_type
        MODEL_TYPE = self.classifier
        MODEL_PATH = self.classifier_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = get_model(MODEL_TYPE, DATA_TYPE, device, MODEL_PATH)
        print("classifier is inited!")
        # classifier = None
        # if MODEL_TYPE == 'resnet50':
        #     classifier = resnet50()
        #     num_ftrs = classifier.fc.in_features
        #     if DATA_TYPE == "cifar10":
        #         classifier.fc = nn.Linear(num_ftrs, 10)
        #     elif DATA_TYPE == "imagenet50":
        #         classifier.fc = nn.Linear(num_ftrs, 50)
        #     classifier.to(device)
        #     classifier.load_state_dict(torch.load(MODEL_PATH + MODEL_TYPE + "_" + DATA_TYPE + ".pth"))
            # print("nice")
        self.netG.eval()
        self.test_metrics.reset()
        self.size = 0
        self.correct = 0
        if self.phase == 'train':
            self.netG.set_new_noise_schedule(phase="test")
            with torch.no_grad():
                for phase_data in tqdm.tqdm(self.val_loader):
                    self.set_input(phase_data)
                    self.size += len(self.gt_image)
                    if self.opt['distributed']:
                        if self.task in ['inpainting', 'uncropping']:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                                                                                     y_0=self.gt_image, mask=self.mask,
                                                                                     sample_num=self.sample_num)
                        else:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image,
                                                                                     sample_num=self.sample_num)
                    else:
                        if self.task in ['inpainting', 'uncropping']:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                                                                                     y_0=self.perturb_image, mask=self.mask,
                                                                                     sample_num=self.sample_num)
                            # tsf = transforms.RandomVerticalFlip(p=1.0)
                            # self.output = tsf(self.output)
                            min_val = torch.min(torch.flatten(self.output))  # 获取整个tensor中的最小值
                            max_val = torch.max(torch.flatten(self.output))  # 获取整个tensor中的最大值

                            print(min_val)  # 打印整个tensor的最小值
                            print(max_val)  # 打印整个tensor的最大值
                            self.evaluate(X=self.output, model=classifier,y= self.label, device=device)
                        else:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                    self.iter += self.batch_size
                    self.writer.set_iter(self.epoch, self.iter, phase='test')
                    for met in self.metrics:
                        key = met.__name__
                        value = met(self.gt_image, self.output)
                        self.test_metrics.update(key, value)
                        self.writer.add_scalar(key, value)
                    for key, value in self.get_current_visuals(phase='test').items():
                        self.writer.add_images(key, value)
                    self.writer.save_images(self.save_current_results())
                self.correct /= self.size
                correct = 100 * self.correct
                self.logger.info(f"test:\n Accuracy:{correct} \n")
                print(f"test:\n Accuracy:{correct} \n")
            test_log = self.test_metrics.result()
            ''' save logged informations into log dict '''
            test_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard '''
            for key, value in test_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))
            # val_end: test_noise_schedule->train_noise_schedule
            self.netG.set_new_noise_schedule(phase=self.phase)
        else:
            with torch.no_grad():
                for phase_data in tqdm.tqdm(self.phase_loader):
                    self.set_input(phase_data)
                    self.size += len(self.gt_image)
                    if self.opt['distributed']:
                        if self.task in ['inpainting', 'uncropping']:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                                                                                     y_0=self.gt_image, mask=self.mask,
                                                                                     sample_num=self.sample_num)
                        else:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                                                                                     y_0=self.gt_image, mask=self.mask,
                                                                                     sample_num=self.sample_num)
                    else:
                        if self.task in ['inpainting', 'uncropping']:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                                                                                     y_0=self.perturb_image, mask=self.mask,
                                                                                     sample_num=self.sample_num)
                            # tsf = transforms.RandomVerticalFlip(p=1.0)
                            # self.output = tsf(self.output)
                            if self.data_type == "mnist":
                                self.output = torch.clamp(self.output, min=0, max=1)
                            min_val = torch.min(torch.flatten(self.output))  # 获取整个tensor中的最小值
                            max_val = torch.max(torch.flatten(self.output))  # 获取整个tensor中的最大值
                            print("yes")
                            print(min_val)  # 打印整个tensor的最小值
                            print(max_val)  # 打印整个tensor的最大值
                            self.evaluate(X=self.output, model=classifier,y= self.label, device=device)
                            mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
                            std = torch.tensor([0.229, 0.224, 0.225]).to(device)
                            self.gt_image = self.gt_image * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
                            self.cond_image = self.cond_image  * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
                            self.gt_image = (self.gt_image - 0.5) * 2
                            self.cond_image = (self.cond_image - 0.5) * 2
                            self.output = (self.output - 0.5) * 2
                            self.adv_img = self.adv_img * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
                            self.adv_img = (self.adv_img - 0.5) * 2
                            # print(self.output.shape)
                        else:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                    self.iter += self.batch_size
                    self.writer.set_iter(self.epoch, self.iter, phase='test')
                    for met in self.metrics:
                        key = met.__name__
                        value = met(self.gt_image, self.output)
                        self.test_metrics.update(key, value)
                        self.writer.add_scalar(key, value)
                    for key, value in self.get_current_visuals(phase='test').items():
                        self.writer.add_images(key, value)
                    if self.only_out == 'cat':
                        self.writer.save_npy(self.save_current_results())
                    else:
                        self.writer.save_images(self.save_current_results())
                self.correct /= self.size
                correct = 100 * self.correct
                self.logger.info(f"test:\n Accuracy:{correct} \n")
                print(f"test:\n Accuracy:{correct} \n")
            test_log = self.test_metrics.result()
            ''' save logged informations into log dict '''
            test_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard '''
            for key, value in test_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def evaluate(self, X, y, model, device):
        # batch = len(X)
        if self.classifier == "wide_resnet":
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).to(device)
            X = (X + 1) * 0.5
            X = (X - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
            min_val = torch.min(torch.flatten(X))  # 获取整个tensor中的最小值
            max_val = torch.max(torch.flatten(X))  # 获取整个tensor中的最大值
            print("得到-1,1的x_re\n")
            print(min_val)  # 打印整个tensor的最小值
            print(max_val)  # 打印整个tensor的最大值
            print("wide_resnet")
        if self.data_type == "imagenet50":
            mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(device)
            X = (X - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
            min_val = torch.min(torch.flatten(X))  # 获取整个tensor中的最小值
            max_val = torch.max(torch.flatten(X))  # 获取整个tensor中的最大值
            print(min_val)  # 打印整个tensor的最小值
            print(max_val)  # 打印整个tensor的最大值
        model.eval()
        # loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            self.pred = model(X).argmax(1)
            compare = (self.pred == y)
            self.correct += compare.sum().item()
        self.logger.info(self.size)
        self.logger.info(f"test:\n Accuracy:{self.correct} \n")
        print(self.size)
        print(f"test:\n Accuracy:{self.correct} \n")
        return self.correct


    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label + '_ema', strict=False)

    def test_n_times(self, n_times):
        for i in range(n_times):
            util.set_seed(i+42)
            self.test()
        # 计算准确度
        correct = 0
        for row in self.n_pred:
            max_value_index = max(range(10), key=lambda i: row[i])  # 获取每行前10列的最大值
            if max_value_index == row[10]:  # 将最大值与第11列进行比较
                correct += 1

        accuracy = correct / len(self.n_pred)
        self.logger.info(f"准确度:{accuracy}")
        print("准确度:", accuracy)


    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label + '_ema')
        self.save_training_state()
