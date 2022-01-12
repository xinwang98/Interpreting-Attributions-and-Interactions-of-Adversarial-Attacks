import torch
import numpy as np
import copy
from torch.autograd import Variable
import os
import csv
import torchvision.transforms as transforms
from codes.util import *
from .cw_attack import *
from dataset import *

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def i_fgsm(net, criterion, x, target, mean, std, mask, alpha=1/256, iteration=80, x_val_min=0, x_val_max=1):
    # h = net((x - mean) / std)
    # _, ori_pred = torch.max(h, dim=1)
    # print(f'Original prediction is {ori_pred.item()}')
    pixel_value = 1 / 256
    for e in range(1, 256):
        x_adv = Variable(x.data, requires_grad=True)
        # x_adv = copy.deepcopy(x)
        # x_adv.requires_grad_()
        eps = e * pixel_value
        for i in range(iteration):
            h_adv = net((x_adv-mean)/std)
            _, adv_pred = torch.max(h_adv, dim=1)
            if adv_pred.item() == target:
                print(f'Attack successful L_i = {eps} step = {i}')
                return 1, eps, x_adv
            cost = criterion(h_adv, target)
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - alpha * x_adv.grad
            x_adv = where(x_adv > x + eps, x + eps, x_adv)
            x_adv = where(x_adv < x - eps, x - eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

            ptb = x_adv - x
            x_adv = x + ptb*mask
            x_adv = Variable(x_adv.data, requires_grad=True)

    return 0, float('inf'), None

