import argparse
import os

import torch

from codes.attack import *
from codes.interaction import *
from codes.util import *
from component_func import celeba_run_func, run_func
from generate_info import run_info

torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser('parameters')

# hardware
parser.add_argument('--gpu_id', type=int, default=3)

# model
parser.add_argument('--model_type', default='res18')
parser.add_argument('--train_type', default='normal')
parser.add_argument('--pretrain', default='pretrained')
# data
parser.add_argument('--data_type', default='tiny_cub')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--padding_size', type=int, default=0)
# shapley
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--super_size', type=int, default=4)
parser.add_argument('--patch_num', type=int, default=8, help='divide image into kxk patches')
parser.add_argument('--seed', type=int, default=1)
# cluster
parser.add_argument('--component_size', type=int, default=4)
parser.add_argument('--cluster_patch_threshold', type=float, default=0.2)
parser.add_argument('--cluster_global_threshold', type=float, default=0.5)
parser.add_argument('--attr', type=int, default=20)

args = parser.parse_args()

# path
args.data_root = f'./data/interaction/{args.data_type}/img'
args.seg_root = f'./data/interaction/{args.data_type}/seg'
args.segpixel_root = f'./data/interaction/{args.data_type}/segpixels'
args.stats_root = f'./data/interaction/stats/{args.data_type}'

args.perturb_root = os.path.join('./experiment/interaction/perturb', f'{args.pretrain}_{args.train_type}/{args.data_type}_{args.model_type}')
args.csv_root = os.path.join(
    f'./experiment/interaction/csv_{args.super_size}',
    f'{args.pretrain}_{args.train_type}/{args.data_type}_{args.model_type}')
args.component_root = f'./experiment/interaction/component_{args.super_size}/{args.pretrain}_{args.train_type}/{args.data_type}_{args.model_type}'
args.info_root = f'./experiment/interaction/info_{args.super_size}/{args.pretrain}_{args.train_type}/{args.data_type}_{args.model_type}'
if args.data_type == 'voc':
    args.num_labels = 20
    args.tlab = 2
elif args.data_type == 'sdd':
    args.num_labels = 120
    args.tlab = 119
elif args.data_type == 'tiny_cub':
    args.num_labels = 10
    args.tlab = -1
elif args.data_type == 'celeba':
    args.num_labels = 80
    args.attrs = [20, 39, 31, 36, 15]
else:
    raise Exception('Invalid data_type')
seed_everything(args.seed)
if args.data_type == 'tiny_cub' or args.data_type == 'voc':
    args.model_path = f'./checkpoint/interaction/{args.pretrain}_{args.train_type}_{args.data_type}_{args.model_type}.bin'
elif args.data_type == 'sdd' or args.data_type == 'celeba':
    args.model_path = f'./checkpoint/interaction/{args.pretrain}_{args.train_type}_{args.data_type}_{args.model_type}.tar'
else:
    raise Exception('Invalid data_type')

if args.model_type == 'res18':
    args.batch_size = 128
elif args.model_type == 'res34':
    args.batch_size = 64
elif args.model_type == 'res50':
    args.batch_size = 32
else:
    raise Exception('Invalid model_type')
if args.super_size == 2:
    args.shapley_compute_threshold = 0.9
    args.stop = 256
elif args.super_size == 4:
    args.shapley_compute_threshold = 0.9
    args.stop = 64
else:
    raise Exception('Invalid super_size')

generate_perturbation(args)
level_cluster(args)
run_func(args)
run_info(args)
