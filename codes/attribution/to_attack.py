import csv
import os

import numpy as np
import torchvision.transforms as transforms
from codes.util import *
from dataset import *

from .cw_attack import *
from .li_fgsm import *


def to_attack(args):
    device = torch.device("cuda:{}".format(args.gpu_id))
    dataset = OriginDataset(args.data_root, args.stats_root, args.data_type, args.img_size, args.padding_size)
    mean, std = dataset.rgb_mean, dataset.rgb_std
    tmean = torch.from_numpy(mean.astype(np.float32)).to(device)
    tstd = torch.from_numpy(std.astype(np.float32)).to(device)

    model = load_checkpoint(args.num_labels, args.model_type, checkpoint_path=args.model_path, device='cpu')
    model.to(device)
    padding_area = 4 * args.padding_size * (args.img_size-args.padding_size)
    block_area = (args.img_size-2*args.padding_size)**2 / (args.partition**2)
    inner = args.img_size - 2*args.padding_size
    for batch_id, (image, label, fname, cate) in enumerate(dataset):
        print (f'current computation label: {label}',fname,  end = ' ')
        csv_root = os.path.join(args.csv_root, 'l2', args.model_type, args.data_type, f'{label}', f'{fname}')
        per_root = os.path.join(args.perturb_root, 'l2', args.model_type, args.data_type, f'{label}', f'{fname}')

        img = image[np.newaxis, :]
        preimg = (img - dataset.rgb_mean) / dataset.rgb_std
        timg = torch.from_numpy(preimg).to(device).float()
        out = model(timg).data.cpu().numpy()
        pred = np.argmax(out)
        print(f'init pred: {pred}')
        if pred != label:
            continue
        pers, c, flag = cw_attack(device=device, img=image, model=model, target=args.tlab,
                           partition=1, batch_size=1, mean=tmean, std=tstd,
                         outer=args.img_size, binary_search_steps=5, max_iterations=1000)
        if flag != 1:
            print ('attack fail')
            continue
        mk_dir(csv_root)
        mk_dir(per_root)
        total = [[],[]]
        for num in range(args.begin, args.high):
            print(f'num_allowed {num}: ', end=' ')
            if num == 2:
                a = 1

            initc = 2*c[0]*args.img_size*args.img_size / (num*block_area+padding_area)
            sample_num = args.sample_num

            csv_path = os.path.join(csv_root, str(num)+'.csv')
            with open(csv_path, 'w') as f:
                csv_write = csv.writer(f)
                for i in range(sample_num//args.batch_size):
                    inds = generate_inds(args, num, total, i)
                    total[i] = inds.copy()
                    masks = generate_masks(inds, args.padding_size, inner, args.partition)
                    prev_inds = []
                    for j in range(len(inds)):
                        prev_inds.append(inds[j].copy())
                    ninds, pers = cw_attack(device=device, img=image, model=model,
                                           batch_size=args.batch_size,
                                         target=args.tlab, mean=tmean, std=tstd,
                                         binary_search_steps=args.bss, max_iterations=args.mir,
                                        learning_rate=args.lr, initial_const=initc, partition=args.partition,
                                           outer=args.img_size,
                                           inds=prev_inds, nmasks=masks)
                    csv_write.writerows(ninds)

        for num in range(args.end, args.high-1, -1):
            print(f'num_allowed {num}: ', end=' ')

            initc = 2 * c[0] * args.img_size * args.img_size / (num * block_area + padding_area)
            sample_num = args.sample_num

            csv_path = os.path.join(csv_root, str(num) + '.csv')
            with open(csv_path, 'w') as f:
                csv_write = csv.writer(f)
                for i in range(sample_num // args.batch_size):
                    inds = generate_inds(args, num, total, i)
                    total[i] = inds.copy()
                    masks = generate_masks(inds, args.padding_size, inner, args.partition)
                    prev_inds = []
                    for j in range(len(inds)):
                        prev_inds.append(inds[j].copy())
                    ninds, pers = cw_attack(device=device, img=image, model=model,
                                            batch_size=args.batch_size,
                                            target=args.tlab, mean=tmean, std=tstd,
                                            binary_search_steps=args.bss, max_iterations=args.mir,
                                            learning_rate=args.lr, initial_const=initc, partition=args.partition,
                                            outer=args.img_size,
                                            inds=prev_inds, nmasks=masks)

                    if i==0 and num == args.end:
                        np.save(f'{per_root}/ptb.npy', pers[0])


def li_attack(args):
    device = torch.device("cuda:{}".format(args.gpu_id))
    dataset = OriginDataset(args.data_root, args.stats_root, args.data_type, args.img_size, args.padding_size)
    mean, std = dataset.rgb_mean, dataset.rgb_std

    model = load_checkpoint(args.num_labels, args.model_type, checkpoint_path=args.model_path, device='cpu')
    model.to(device)
    tmean = torch.from_numpy(mean.astype(np.float32)).to(device)
    tstd = torch.from_numpy(std.astype(np.float32)).to(device)
    tlab = torch.tensor([args.tlab]).to(device)
    criterion = nn.CrossEntropyLoss()
    inner = args.img_size - 2*args.padding_size
    for batch_id, (image, label, fname, cate) in enumerate(dataset):

        csv_root = os.path.join(args.csv_root, 'li', args.model_type, args.data_type, f'{label}', f'{fname}')
        per_root = os.path.join(args.perturb_root, 'li', args.model_type, args.data_type, f'{label}', f'{fname}')

        print (f'current computation label: {label}')
        image = torch.from_numpy(image).unsqueeze(0).float().to(device)

        mk_dir(csv_root)
        mk_dir(per_root)
        total = [[], []]
        for num in range(args.begin, args.high):
            print(f'num_allowed {num}: ')

            sample_num = args.sample_num

            csv_path = os.path.join(csv_root, str(num) + '.csv')
            with open(csv_path, 'w') as f:
                csv_write = csv.writer(f)
                for i in range(sample_num // args.batch_size):
                    inds = generate_inds(args, num, total, i)
                    total[i] = inds.copy()
                    masks = generate_masks(inds, args.padding_size, inner, args.partition)
                    pers = []
                    for j in range(len(inds)):
                        print (j, end=' ')
                        res = inds[j].copy()
                        mask = torch.from_numpy(masks[j]).float().to(device)

                        success, li_dist, adv_image = i_fgsm(model, criterion, image, tlab, tmean, tstd, mask)
                        res.append(li_dist)
                        res.append(success)
                        csv_write.writerow(res)

        for num in range(args.end, args.high - 1, -1):
            print(f'num_allowed {num}: ', end=' ')

            sample_num = args.sample_num

            csv_path = os.path.join(csv_root, str(num) + '.csv')
            with open(csv_path, 'w') as f:
                csv_write = csv.writer(f)
                for i in range(sample_num // args.batch_size):
                    inds = generate_inds(args, num, total, i)
                    total[i] = inds.copy()
                    masks = generate_masks(inds, args.padding_size, inner, args.partition)
                    pers = []
                    for j in range(len(inds)):
                        print (j, end=' ')
                        res = inds[j].copy()
                        mask = torch.from_numpy(masks[j]).float().to(device)

                        success, li_dist, adv_image = i_fgsm(model, criterion, image, tlab, tmean, tstd, mask)
                        res.append(li_dist)
                        res.append(success)
                        pers.append(adv_image)
                        csv_write.writerow(res)

                        if i==0 and j==0 and num==args.end:
                            ptb = (adv_image - image).data.cpu().numpy()
                            np.save(f'{per_root}/ptb.npy', ptb)
