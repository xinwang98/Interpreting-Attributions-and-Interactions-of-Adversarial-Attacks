import random

import numpy as np


def generate_masks(inds, pad, inner, a):
    outer = inner + 2*pad
    b = int(inner / a)   # 区域大小
    out_masks = np.ones((64, 3, outer, outer))
    in_masks = np.zeros((64, 3, inner, inner))
    for i in range(64):
        ind = inds[i]   # 1D向量
        for j in ind:
            x = int(j // a)
            y = int(j % a)
            in_masks[i:i+1, :, b*x:b*x+b, b*y:b*y+b] = 1
    out_masks[:, :, pad:outer-pad, pad:outer-pad] = in_masks
    out_masks.astype(np.float32)
    return out_masks  # 2D，4D


def generate_inds(args, num, total, turn):
    src = [i for i in range(64)]
    if num == 0:
        inds = []
        for i in range(64):
            inds.append([])
    elif num == 1:
        inds = [[i] for i in range(64)]
    elif 1 < num < args.low:
        inds = total[turn].copy()
        for i in range(len(inds)):
            avail = []
            for j in range(64):
                if j not in inds[i]:
                    avail.append(j)
            id = random.sample(avail, 1)
            inds[i].append(id[0])
    elif args.low <= num <= args.high:
        inds = []
        for i in range(64):
            ind = random.sample(src, num)  # 1D向量
            inds.append(ind)
    elif num == 64:
        inds = []
        for i in range(64):
            ind = [j for j in range(64)]
            inds.append(ind)
    elif num == 63:
        inds = total[turn].copy()
        for i in range(64):
            inds[i].remove(i)
    else:
        inds = total[turn].copy()
        for i in range(64):
            id = random.sample(total[turn][i], 1)
            inds[i].remove(id[0])
    return inds
