import torch
import numpy as np
from codes.util import load_checkpoint
from dataset import OriginDataset
import csv
import os

def generate_mask(ids, img_size, super_size):
    super_imgsize = img_size // super_size
    mask = np.ones((3, img_size, img_size))
    rows = ids // super_imgsize
    cols = ids % super_imgsize
    for i in range(super_size):
        for j in range(super_size):
            mask[:, super_size * rows+i, super_size * cols+j] = 0

    center = (rows.mean(), cols.mean())
    return mask, center


def run_func(args):
    device = torch.device("cuda:{}".format(args.gpu_id))
    dataset = OriginDataset(args.data_root, args.stats_root, args.data_type, args.img_size, args.padding_size)

    model = load_checkpoint(
        args.num_labels,
        args.model_type,
        checkpoint_path=args.model_path,
        device='cpu',
        train_type=args.train_type,
        pretrain=args.pretrain).to(device)

    mean, std = dataset.rgb_mean, dataset.rgb_std
    if args.train_type == 'adv':
        mean, std = np.zeros(mean.shape), np.ones(std.shape)
    for batch_id, (image, label, fname, category) in enumerate(dataset):
        ptb_path = os.path.join(args.perturb_root, category, fname, f'ptb.npy')
        clus_path = os.path.join(args.csv_root, category, fname,
                                 f'clusback{args.stop}.csv')

        adv_info = np.load(ptb_path, allow_pickle=True).item()
        ptb, target = adv_info['ptb'], adv_info['tgt']
        print(batch_id, category, fname)

        img = (image + ptb)[np.newaxis, :]
        preimg = (img - mean) / std
        timg = torch.from_numpy(preimg).to(device).float()
        out = model(timg).data.cpu().numpy().squeeze()
        yl, yt = out[label], out[target]

        print('adv:', yl, yt)

        clusters = np.genfromtxt(clus_path, delimiter=',').astype(int)
        if len(clusters.shape)<2:
            clusters = clusters[np.newaxis,:]
        clusters = clusters[:, 1:]
        for i in range(clusters.shape[0]):
            clus = clusters[i]
            mask, center = generate_mask(clus, args.img_size, args.super_size)
            cptb = mask * ptb
            cimg = (image + cptb)[np.newaxis, :]
            pimg = (cimg - mean) / std
            timg = torch.from_numpy(pimg).to(device).float()
            out = model(timg).data.cpu().numpy().squeeze()
            print(center, out[label] - yl, out[target] - yt)

