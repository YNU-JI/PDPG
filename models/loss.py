import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.autograd import Variable
# import pytorch_ssim



# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)

def l1_loss(output, target):
    loss_fn = nn.L1Loss(reduction='sum')
    return loss_fn(output, target)


def tv_loss(img):
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy)



def ssim(output, target, window_size=11, size_average=True):
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    channel = output.size()[1]
    window = torch.Tensor(gaussian(window_size, 1.5)).unsqueeze(0).unsqueeze(0)
    window = window.repeat(channel, 1, 1, 1)
    if output.is_cuda:
        window = window.cuda(output.device.index)
    mu1 = F.conv2d(output, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(output * output, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(output * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return torch.mean(ssim_map)
    else:
        return torch.sum(ssim_map)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def ssim_loss(output, target):
    return  1 - ssim(output, target)

def cosine(output, target):
    output = output.view(-1)
    target = target.view(-1)
    cosine_similarity = F.cosine_similarity(output, target, dim=0)
    return cosine_similarity

def control_loss(output, target, adv_input=None, fir=None, fir_adv=None, choice='control_mse'):
    if choice == 'l1_loss':
        loss = l1_loss(output, target)
        print(loss)
        return loss
    elif choice == 'mse':
        return F.mse_loss(output, target)
    elif choice == 'control_mse' and adv_input is not None:
        lamda = 10
        # if torch.allclose(target, adv_input):
        #     1/0
        loss1 = F.mse_loss(output, target)
        loss2 = F.mse_loss(target, adv_input)
        loss3 = F.mse_loss(fir, fir_adv)
        print(f"\nloss1:{loss1}\nloss2:{loss2}\nloss3:{loss3}")
        return loss1 + lamda * loss2
    elif choice == 'control_smooth_l1_loss' and adv_input is not None:
        lamda1 = -0.1
        lamda2 = 0.1
        # if torch.allclose(target, adv_input):
        #     1/0
        loss1 = F.smooth_l1_loss(output, target)
        loss2 = F.smooth_l1_loss(target, adv_input)
        loss3 = F.smooth_l1_loss(fir, fir_adv)
        # print(f"target_loss:{loss1}\nadv_loss:{loss2}")
        # loss3 = F.smooth_l1_loss(output, adv_input)
        print(f"\nloss1:{loss1}\nloss2:{loss2}\nloss3:{loss3}")

        return loss1 +  0.5*loss2
    elif choice == 'control_l1' and adv_input is not None:
        loss1 = l1_loss(output, target)
        loss2 = l1_loss(target, adv_input)
        loss3 = l1_loss(fir, fir_adv)
        lamda = 1
        print(f"\nloss1:{loss1}\nloss2:{loss2}\nloss3:{loss3}")
        return loss1 +1* loss2
    elif choice == 'cosine':
        loss1 = -cosine(output, target)
        loss2 = -cosine(target, adv_input)
        loss3 = -cosine(fir, fir_adv)
        loss = loss1 + loss2 + loss3
        print(f"\nloss1:{loss1}\nloss2:{loss2}\nloss3:{loss3}")
        return loss
    elif choice == 'mix_loss' and adv_input is None:
        loss_fn = nn.L1Loss(reduction='mean')
        loss1 = loss_fn(output, target)
        loss2 = mse_loss(output, target)
        print(f"mae: {loss1}\n mse: {loss2}")
        return loss1 + loss2
    else:
        # print(output.shape, target.shape)
        marg = abs(output - target)
        derta = 1e-3
        count_derta = torch.sum(marg > derta)
        marg_mean = marg.mean()
        print(f"marg.mean:{marg_mean} , count_derta: {count_derta}, total:{torch.numel(marg)}, per:{count_derta/torch.numel(marg)}\n")
        count = torch.sum(marg > 1)
        total = torch.numel(marg)
        percentage = (count / total) * 100
        min_val = torch.min(torch.flatten(marg))  # 获取整个tensor中的最小值
        max_val = torch.max(torch.flatten(marg))  # 获取整个tensor中的最大值
        loss1 = torch.mean((output - target) ** 2)
        # output[marg < derta] = 0
        # target[marg < derta] = 0
        # output[marg > 2 * marg_mean] = output[marg > 2 * marg_mean] * 2
        # target[marg > 2 * marg_mean] = target[marg > 2 * marg_mean] * 2
        # loss2 = mse_loss(output, target)
        print(f"min: {min_val}, max: {max_val}, per: {percentage} , mse_loss: {loss1}\n")
        return F.smooth_l1_loss(output, target)


def weight_loss(output, target):
    mse_loss = F.mse_loss(output, target)
    ssim_loss = 1 - ssim(output, target)
    return 0.8 * mse_loss + 0.2 * ssim_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

