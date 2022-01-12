# coding=utf-8

from __future__ import division

import logging
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.utils.data.dataloader as Data
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import pdb
# pixel mean dis 0.0006


def cw_attack(device, img, label, model, target, mean, std, num_labels=20, bounds=(0, 1),
              confidence=0, max_iterations=1000, learning_rate=5e-3, initial_const=1e-2, binary_search_steps=5):
    model = model.to(device)
    boxmin, boxmax = bounds[0], bounds[1]
    # c的初始化边界
    lower_bound = 0
    const = initial_const
    upper_bound = 1e10
    # the best l2, score, and image attack
    o_bestl2 = 1e10
    o_bestscore = -1
    o_bestattack = None
    # the resulting image, tanh'd to keep bounded from boxmin to boxmax
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.
    for outer_step in range(binary_search_steps):
        # 把原始图像转换成图像数据和扰动的形态
        timg = np.arctanh((img-boxplus)/boxmul*0.999999).astype(np.float32)
        timg = (torch.from_numpy(timg)).to(device)

        modifier = torch.zeros_like(timg).to(device).float()
        modifier.requires_grad = True
        # 定义优化器 仅优化modifier
        optimizer = torch.optim.Adam([modifier], lr=learning_rate)

        for iteration in range(1, max_iterations + 1):
            optimizer.zero_grad()
            # 定义新输入
            newimg = torch.tanh(modifier + timg) * boxmul + boxplus
            image = torch.from_numpy(img).to(device).float()

            pnewimg = ((newimg - mean) / std).unsqueeze(0)
            output = model(pnewimg)[0]
            sc = output.data.cpu().numpy()

            # 定义cw中的损失函数
            loss2 = torch.dist(newimg, image, p=2) ** 2
            if target >= 0:
                pred = np.argmax(sc)
                tmp = output[pred] - output[target] + confidence
            else:
                sc[label] = -1e10
                pred = np.argmax(sc)
                tmp = -output[pred] + output[label] + confidence
            loss1 = const * torch.clamp(tmp, min=0)
            loss = loss1 + loss2

            loss.backward(retain_graph=True)
            optimizer.step()
            l2 = loss2
            # print (pred, modifier.norm())

            if target >= 0:
                if (pred == target) and (l2 < o_bestl2):
                    o_bestl2 = l2
                    o_bestscore = pred
                    o_bestattack = newimg.data.cpu().numpy()
            else:
                if (output[pred] > output[label]) and (l2 < o_bestl2):
                    o_bestl2 = l2
                    o_bestscore = pred
                    o_bestattack = newimg.data.cpu().numpy()
        # pdb.set_trace()
        if target >= 0:
            if o_bestscore == target:
                upper_bound = min(upper_bound, const)
            else:
                lower_bound = max(lower_bound, const)
            if upper_bound < 1e9:
                const = (lower_bound + upper_bound) / 2
            else:
                const *= 10

        else:
            if o_bestscore>=0 and o_bestscore != label:
                upper_bound = min(upper_bound, const)
            else:
                lower_bound = max(lower_bound, const)
            if upper_bound < 1e9:
                const = (lower_bound + upper_bound) / 2
            else:
                const *= 10

    return o_bestattack, o_bestscore
