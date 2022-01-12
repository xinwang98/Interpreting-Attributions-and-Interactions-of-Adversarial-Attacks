import os
import csv
import numpy as np
import argparse
from codes.util import load_checkpoint, mk_dir
from codes.interaction import cluster, patch_shap, global_shap
import torch
from dataset import OriginDataset
import time

def level_cluster(args):
    device = torch.device("cuda:{}".format(args.gpu_id))
    dataset = OriginDataset(args.data_root, args.stats_root, args.data_type, args.img_size, args.padding_size)
    model = load_checkpoint(
        args.num_labels,
        args.model_type,
        checkpoint_path=args.model_path,
        device='cpu',
        train_type=args.train_type, pretrain=args.pretrain).to(device)
    mean, std = dataset.rgb_mean.astype(np.float32), dataset.rgb_std.astype(np.float32)
    if args.train_type == 'adv' or 'no' in args.pretrain:
        mean, std = np.zeros(mean.shape).astype(np.float32), np.ones(std.shape).astype(np.float32)
    patch_size = args.img_size // args.patch_num
    for batch_id, (image, label, fname, category) in enumerate(dataset):

        ptb_path = os.path.join(args.perturb_root, category, fname, f'ptb.npy')

        adv_info = np.load(ptb_path, allow_pickle=True).item()
        ptb, target = adv_info['ptb'], adv_info['tgt']
        save_path = os.path.join(args.csv_root, category, fname)
        mk_dir(save_path)
        print(category, fname)
        tgt = 1
        while tgt <= args.component_size:
            begin = time.time()
            method = f'backshap{tgt}'

            to_save = patch_shap(
                model=model,
                label=label,
                image=image,
                perturbation=ptb,
                rgb_mean=mean,
                rgb_std=std,
                target=target,
                device=device,
                seed=args.seed,
                shapley_method=method,
                batch_size=args.batch_size,
                component_size=args.component_size,
                super_size=args.super_size,
                patch_size=patch_size,
                patch_num=args.patch_num,
                sample_num=args.sample_num)
            with open(f'{save_path}/{method}.csv', 'w') as f:
                wrt = csv.writer(f)
                wrt.writerows(to_save)
            end = time.time()
            print(f'component size {tgt}, time cost {(end-begin):.2f}')
            tgt *= args.component_size
        cluster(save_path, tgt=args.component_size, patch_num=args.patch_num, component_size=args.component_size, patch_threshold=args.cluster_patch_threshold, global_threshold=args.cluster_global_threshold)

        while tgt <= args.stop:
            begin = time.time()
            method = f'backshap{tgt}'
            to_save = global_shap(
                clus_path=save_path,
                model=model,
                label=label,
                image=image,
                perturbation=ptb,
                rgb_mean=mean,
                rgb_std=std,
                target=target,
                device=device,
                seed=args.seed,
                shapley_method=method,
                batch_size=args.batch_size,
                img_size=args.img_size,
                patch_size=patch_size,
                patch_num=args.patch_num,
                super_size=args.super_size,
                sample_num=args.sample_num,
                threshold=args.shapley_compute_threshold,
                component_size=args.component_size)
            with open(f'{save_path}/{method}.csv', 'w') as f:
                wrt = csv.writer(f)
                wrt.writerows(to_save)
            cluster(save_path, tgt=tgt, patch_num=args.patch_num, component_size=args.component_size, patch_threshold=args.cluster_patch_threshold, global_threshold=args.cluster_global_threshold)
            end = time.time()
            print(f'component size {tgt}, time cost {(end-begin):.2f}')
            tgt = args.component_size * tgt
        # break