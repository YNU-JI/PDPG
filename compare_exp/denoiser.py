import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


def mse_loss(real_image, fake_image):
    return F.mse_loss(real_image, fake_image,reduction='mean')

def l1_loss(real_image, fake_image):
    loss_fn = nn.L1Loss(reduction='mean')
    return loss_fn(real_image, fake_image)


class Denoiser():
    """
    conf : 配置项，包括图片HWC,EPOCH,BATCH_SIZE,lr,beta1
    rec_w 超参数，控制损失重要性
    """
    def __init__(self, conf, unet, classifier=None):
        self.loss_total = None
        self.batch_num = 0
        self.denoised_correct = 0.0
        self.x_correct = 0.0
        self.denoiser = unet
        # self.inputH = conf.imageH
        # self.inputW = conf.imageW
        # self.channel = conf.channel
        self.epochs = conf["epochs"]
        # self.batch_size = conf.batch_size
        self.optimizer = optim.Adam(self.denoiser.parameters(), lr=1e-3, betas=(0.5, 0.999), eps=1e-4, amsgrad=False)
        self.rec_w = conf["rec_w"]
        # self.denoiser = denoiser
        self.classifier = classifier
        # print(self.denoiser)
        # print(self.classifier)

    def start_training(self, inputs, lables, method):
        self.batch_num += 1
        self.denoiser.train()
        self.classifier.eval()
        if method == 'DAE':
            # 文章 Towards deep neural network architectures robust to adversarial examples,思想：使用自编码重构输入期望去除扰动
            # dataset = TensorDataset(x, x_adv)
            # dataloader = DataLoader(dataset, batch_size=self.batch_size)
            loss = self.train_denoiser_DAE(inputs=inputs, labels=lables)
            if self.batch_num % 100 == 0:
                print(f"training_loss: {loss:>7f}")

        if method == 'HGD':
            # 文章 defense against adversarial attacks using high-level representation guided denoiser的思想，采用logits比对
            loss = self.train_denoiser_HGD(inputs=inputs, labels=lables)
            if self.batch_num % 100 == 0:
                print(f"training_loss: {loss:>7f}")

        if method == 'TD':
            # 文章Transferable Adversarial Defense by Fusing Reconstruction Learning and Denoising Learning,目的增强可迁移特性
            loss_y, loss_rec, loss_total = self.train_denoiser_TD(inputs=inputs, labels=lables)
            if self.batch_num % 100 == 0:
                print(f"loss_y: {loss_y:>7f}, loss_rec:{loss_rec:>7f}, loss_total:{loss_total:7f}")
        self.optimizer.zero_grad()
        self.loss_total.backward()
        self.optimizer.step()


    def train_denoiser_DAE(self, inputs, labels):
        x_rec = self.denoiser(inputs)
        self.loss_total = mse_loss(labels, x_rec)
        # loss_total.backward()
        # self.optimizer.step()

        return self.loss_total.item()

    def train_denoiser_HGD(self, inputs, labels):
        x_rec = self.denoiser(inputs)

        y_de_x = self.classifier(x_rec)
        y_de_l = self.classifier(labels)

        self.loss_total = mse_loss(y_de_l, y_de_x)

        # loss_total.backward()
        # self.optimizer.step()

        return self.loss_total.item()

    def train_denoiser_TD(self, inputs, labels):
        x_rec = self.denoiser(inputs)
        y_de_x = self.classifier(x_rec)
        y_de_l = self.classifier(labels)
        loss_y = mse_loss(y_de_l, y_de_x)
        loss_rec = mse_loss(labels, x_rec)
        self.loss_total = loss_y + 0.001 * loss_rec

        # loss_total.backward()
        # self.optimizer.step()

        return loss_y.item(), loss_rec.item(), self.loss_total.item()


    def evaluate(self, inputs, labels, method, y):
        self.batch_num += 1
        self.denoiser.eval()
        self.classifier.eval()
        with torch.no_grad():
            # if method == 'DAE':
            #     loss = self.train_denoiser_DAE(inputs, labels)
            # elif method == 'HGD':
            #     loss = self.train_denoiser_HGD(inputs, labels)
            # elif method == 'TD':
            #     loss = self.train_denoiser_TD(inputs, labels)
            # else:
            #     print("error")
            x_rec = self.denoiser(labels)
            adv_rec = self.denoiser(inputs)
            x_pred = self.classifier(x_rec)
            denoised_pred = self.classifier(adv_rec)
            # test_loss += loss_fn(pred, y).item()
            self.x_correct += (x_pred.argmax(1) == y).sum().item()
            self.denoised_correct += (denoised_pred.argmax(1) == y).sum().item()
            # if self.batch_num % 10 == 0:
            #     print(f"test_loss: {loss}")

