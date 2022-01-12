# coding=utf-8

from __future__ import division
import logging
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision import models
from .mask import *
import math
import time


# 末行参数不可改变
def cw_attack(device, img, model, target, mean, std, outer, inds=None, nmasks=None,
              batch_size=1, partition=8, bounds=(0, 1),
              confidence=5, max_iterations=1000, learning_rate=5e-3, initial_const=1e-2, binary_search_steps=5):
    func_start = time.time()

    model = model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    boxmin, boxmax = bounds[0], bounds[1]
    # c的初始化边界
    lower_bound = torch.tensor([0] * batch_size).float().to(device)
    const = torch.tensor([initial_const] * batch_size).float().to(device)
    upper_bound = torch.tensor([1e10] * batch_size).float().to(device)
    # the best l2, score
    o_bestl2 = 1e10 * torch.ones((batch_size)).float().to(device)
    o_bestpers = torch.zeros(batch_size, 3, outer, outer).to(device).float()
    # the resulting image, tanh'd to keep bounded from boxmin to boxmax
    boxmul = (boxmax - boxmin) / 2.
    boxplus = (boxmin + boxmax) / 2.

    aimg = np.arctanh((img - boxplus) / boxmul * 0.999999).astype(np.float32)
    timg = (torch.from_numpy(aimg)).to(device)
    image = torch.from_numpy(img).to(device).float()
    success = -1 * np.ones(batch_size)
    for outer_step in range(binary_search_steps):
        bestscore = -1 * np.ones(batch_size)
        modifiers = torch.zeros(batch_size, 3, outer, outer).to(device).float()
        modifiers.requires_grad = True
        optimizer = torch.optim.Adam([modifiers], lr=learning_rate)
        for iteration in range(1, max_iterations + 1):
            optimizer.zero_grad()
            newimgs = torch.tanh(modifiers + timg) * boxmul + boxplus
            pers = newimgs - image
            if partition > 1:
                masks = torch.tensor(nmasks).float().to(device)
                pers = pers * masks
                newimgs = image + pers
            pnewimgs = ((newimgs - mean) / std)
            outputs = model(pnewimgs)
            other, preds = torch.max(outputs, dim=1) # prediction val and idx
            loss2 = (torch.norm(pers, p=2, dim=(1, 2, 3)) ** 2).to(device)
            tmp = other - outputs[:,target] + confidence
            loss1 = (torch.clamp(tmp, min=0) * const).to(device)
            loss = (loss1 + loss2).sum()
            loss.backward()
            optimizer.step()

            for i in range(batch_size):
                if (preds[i] == target):
                    success[i] = 1
                    bestscore[i] = 1
                    if loss2[i] < o_bestl2[i]:
                        o_bestl2[i] = loss2[i]
                        o_bestpers[i] = pers[i]

        for i in range(batch_size):
            if target >= 0:
                if bestscore[i] == 1:
                    upper_bound[i] = min(upper_bound[i], const[i])
                else:
                    lower_bound[i] = max(lower_bound[i], const[i])
                if upper_bound[i] < 1e9:
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    const[i] *= 10

    if partition > 1:
        for i in range(batch_size):
            inds[i].sort()
            l2 = float(math.sqrt(o_bestl2[i].data.cpu()))
            inds[i].append(l2)
            inds[i].append(success[i])
    else:
        print ('initial attack l2:', math.sqrt(o_bestl2[0].data.cpu()))
    o_bestpers = o_bestpers.data.cpu().numpy()
    func_end = time.time()
    print ('batch_time {:.3f}'.format(func_end-func_start))

    if partition > 1:
        return inds, o_bestpers
    else:
        return o_bestpers, const.data.cpu().numpy(), success[0]




