import argparse
import os

import torch

from codes.attribution import *
from codes.util import *

torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser('parameters')

# hardware
parser.add_argument('--gpu_id', type=int, default=0)

# model
parser.add_argument('--model_type', default='res18')

# data
parser.add_argument('--data_type', default='cub')
parser.add_argument('--img_size', type=int, default=112)
parser.add_argument('--padding_size', type=int, default=8)

# attack
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--partition', type=int, default=8)
parser.add_argument('--confidence', type=int, default=0)
parser.add_argument('--bss', type=int, default=5, help='binary_search_step')
parser.add_argument('--mir', type=int, default=200, help='max_iterations')
parser.add_argument('--lr', type=float, default=25e-3, help='learning_rate')

# sample
parser.add_argument('--low', type=int, default=10)
parser.add_argument('--high', type=int, default=54)     # sequential sample range
parser.add_argument('--sample_num', type=int, default=64, help='sample times for each case')
parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=64)
parser.add_argument('--times', default='0', help='experiment times')
parser.add_argument('--attack_type', default='l2')
parser.add_argument('--tlab', type=int, default=9)
args = parser.parse_args()

if args.data_type == 'voc':
    args.num_labels = 20
    args.tlab = 2
    args.label = [0,3,6,7,8,12,15,16,18,19]
elif args.data_type == 'cub':
    args.num_labels = 200
    args.tlab = 199
    args.label = [0,23,40,72,99,107,131,148,175,194]
elif args.data_type == 'sdd':
    args.num_labels = 120
    args.tlab = 119
    args.label = [1,2,15,26,31,66,95,96,113,116]


# path
args.data_root = f'./data/attribution/{args.data_type}'
args.stats_root = f'./data/attribution/stats/{args.data_type}'
args.perturb_root = os.path.join(f'./experiment/attribution_{args.tlab}', 'Perturb')
args.csv_root = os.path.join(f'./experiment/attribution_{args.tlab}', 'CSV')
args.show_root = os.path.join(f'./experiment/attribution_{args.tlab}', 'Show')
args.log_root = os.path.join(f'./experiment/attribution_{args.tlab}', 'Log')

args.model_path = f'./checkpoint/attribution/{args.data_type}_{args.model_type}_112.tar'
seed_everything(args.seed)
if args.attack_type == 'l2':
    to_attack(args)
elif args.attack_type == 'li':
    li_attack(args)
else:
    raise Exception('Invalid attack_type')





