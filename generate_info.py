import torch
import numpy as np
import argparse
from codes.util import load_checkpoint, mk_dir
from dataset import OriginDataset
import csv
import os
from PIL import Image


def generate_mask(ids, img_size):
    mask = np.ones((3, img_size, img_size))
    rows = ids // img_size
    cols = ids % img_size
    mask[:, rows, cols] = 0
    center = (rows.mean(), cols.mean())
    return mask, center


def generate_component(args):
    print('--- generate_component ---')
    super_imgsize = args.img_size // args.super_size
    for cate in (os.listdir(args.csv_root)):
        path = os.path.join(args.csv_root, cate)
        # lbl = format(int(lbl),'03d')
        for img in os.listdir(path):
            pixels = {}
            csv_path = os.path.join(path, img, f'clusback{args.stop}.csv')
            if not os.path.exists(csv_path):
                continue
            csv_file = open(csv_path, 'r')
            reader = csv.reader(csv_file)
            for line in reader:
                pixels[float(line[0])] = []
                super_pixels = line[1:]
                for sp in super_pixels:
                    sp_id = int(sp)
                    row, col = args.super_size * (
                        sp_id // super_imgsize), args.super_size * (
                            sp_id % super_imgsize)
                    for i in range(args.super_size):
                        for j in range(args.super_size):
                            pixels[float(line[0])].append(args.img_size *
                                                          (i + row) + col + j)
            save_root = f'{args.component_root}/{cate}'
            mk_dir(save_root)
            np.save(os.path.join(save_root, img[:-4] + '.npy'), pixels)



def generate_segpixels(args):
    print('--- generate segpixels ---')
    for cate in os.listdir(args.component_root):
        comp_path = os.path.join(args.component_root, cate)
        for img in os.listdir(comp_path):
            segpixels_path = os.path.join(args.segpixel_root, cate)
            mk_dir(segpixels_path)
            seg_path = os.path.join(args.seg_root, cate, img[:-4] + '.png')
            if not os.path.exists(seg_path):
                continue
            seg_img = np.array(
                    Image.open(seg_path).convert('RGB').resize(
                        (args.img_size, args.img_size),
                        Image.ANTIALIAS)) / 255.0
            img2d = seg_img.sum(-1)
            img2d[img2d > 0] = 1
            rows, cols = np.where(img2d == 1)
            pixels = args.img_size * rows + cols
            np.save(segpixels_path + '/' + img, pixels)




def cal_score(args):
    print('--- generate info ---')
    device = torch.device("cuda:{}".format(args.gpu_id))
    dataset = OriginDataset(args.data_root, args.stats_root, args.data_type, args.img_size, args.padding_size)

    model = load_checkpoint(
        args.num_labels,
        args.model_type,
        checkpoint_path=args.model_path,
        device='cpu',
        train_type=args.train_type, pretrain=args.pretrain).to(device)

    mean, std = dataset.rgb_mean, dataset.rgb_std
    if args.train_type == 'adv':
        mean, std = np.zeros(mean.shape), np.ones(std.shape)
    for batch_id, (image, label, fname, category) in enumerate(dataset):
        ptb_path = os.path.join(args.perturb_root, category, fname, f'ptb.npy')
        if not os.path.exists(ptb_path):
            continue

        adv_info = np.load(ptb_path, allow_pickle=True).item()
        ptb, target = adv_info['ptb'], adv_info['tgt']
        adv = (image + ptb)[np.newaxis, :]

        info = {}
        preadv = (adv - mean) / std
        tadv = torch.from_numpy(preadv).to(device).float()
        out = model(tadv).data.cpu().numpy().squeeze()
        yl, yt = out[label], out[target]

        info['adv_score'] = {'label': yl.item(), 'target': yt.item()}

        seg_pixels = np.load(
            os.path.join(args.segpixel_root, category, fname[:-4] + '.npy'),
            allow_pickle=True)
        comp_pixels = np.load(
            os.path.join(args.component_root, category, fname[:-4] + '.npy'),
            allow_pickle=True).item()
        seg_set = set(seg_pixels)
        all_comp = []
        for k in comp_pixels.keys():
            comp_set = set(comp_pixels[k])
            # print (len(comp_set))
            fore = seg_set.intersection(comp_set)
            mask, center = generate_mask(np.array(comp_pixels[k]), args.img_size)
            cptb = mask * ptb
            cimg = (image + cptb)[np.newaxis, :]
            pimg = (cimg - mean) / std
            timg = torch.from_numpy(pimg).to(device).float()
            out = model(timg).data.cpu().numpy().squeeze()
            info[center] = {
                'label': out[label],
                'target': out[target],
                'fore': float(len(fore) / len(comp_set))
            }
            all_comp += comp_pixels[k]
        mask, center = generate_mask(np.array(all_comp), args.img_size)
        cptb = mask * ptb
        cimg = (image + cptb)[np.newaxis, :]
        pimg = (cimg - mean) / std
        timg = torch.from_numpy(pimg).to(device).float()
        out = model(timg).data.cpu().numpy().squeeze()
        info['all'] = {
            'label': out[label],
            'target': out[target],
        }

        save_root = os.path.join(args.info_root, category)
        mk_dir(save_root)
        np.save(save_root + f'/{fname[:-4]}.npy', info)
        print(category, fname, 'adv:', yl, yt)


def fore_label(args, rate):
    selected = []
    for cate in os.listdir(
            f'./experiment/interaction/info_{args.super_size}/{args.pretrain}_adv/{args.data_type}_{args.model_type}'
    ):
        for img in os.listdir(
                f'./experiment/interaction/info_{args.super_size}/{args.pretrain}_adv/{args.data_type}_{args.model_type}/{cate}'
        ):
            if os.path.exists(
                    f'./experiment/interaction/info_{args.super_size}/{args.pretrain}_normal/{args.data_type}_{args.model_type}/{cate}'
            ) and img in os.listdir(
                    f'./experiment/interaction/info_{args.super_size}/{args.pretrain}_normal/{args.data_type}_{args.model_type}/{cate}'
            ):
                selected.append(cate + img)
            else:
                print('not used', cate, img)
    for train in ['normal', 'adv']:
        info_root = f'./experiment/interaction/info_{args.super_size}/{args.pretrain}_{train}/{args.data_type}_{args.model_type}'
        comp, fore, dec_label, norm_dec_label, imgs = 0, 0, 0, 0, 0
        fore_pixel, total_pixel = 0, 0
        img_list = []
        for cate in os.listdir(info_root):
            for img in os.listdir(os.path.join(info_root, cate)):
                if cate + img not in selected:
                    continue

                f = np.load(
                    os.path.join(info_root, cate, img),
                    allow_pickle=True).item()
                # print (f.keys())
                adv_label, adv_tgt = f['adv_score']['label'], f[
                    'adv_score']['target']
                no_comp_label, no_comp_tgt = f['all']['label'], f['all'][
                    'target']
                for center in f.keys():
                    # print (center)

                    if isinstance(center, str):
                        # print ('Invalid center')
                        continue
                    comp += 1
                    # print('jhjkf')
                    if f[center]['fore'] > rate:
                        fore += 1
                    fore_pixel +=  f[center]['fore']
                    total_pixel += 1
                    label, tgt = f[center]['label'], f[center]['target']
                    if abs(label - adv_label) > abs(tgt - adv_tgt):
                        dec_label += 1
                    if abs(label - adv_label
                            ) / abs(no_comp_label - adv_label) > abs(
                                tgt - adv_tgt) / abs(no_comp_tgt - adv_tgt):
                        norm_dec_label += 1
                imgs += 1
                img_list.append(img)
                print(img, fore, dec_label, comp)
        print(
            train,
            f'imgs: {imgs}, fore: {fore/comp:.3f}, dec_label: {dec_label/comp:.3f}, fore_pixel: {fore_pixel/total_pixel:.3f}, norm_dec_label: {norm_dec_label/comp:.3f}'
        )

def run_info(args):
    for train in ['normal', 'adv']:
        args.train_type = train
        generate_component(args)
        generate_segpixels(args)
        cal_score(args)
    fore_label(args, rate=0.5)
