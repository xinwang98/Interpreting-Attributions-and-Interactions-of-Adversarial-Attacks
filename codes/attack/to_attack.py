import os
import csv
import numpy as np
from codes.util import load_checkpoint, mk_dir
from dataset import OriginDataset
from codes.attack.cw_attack import cw_attack
import torch

def generate_perturbation(args):
    device = torch.device("cuda:{}".format(args.gpu_id))
    model = load_checkpoint(args.num_labels, args.model_type, checkpoint_path=args.
    model_path, device='cpu', train_type=args.train_type, pretrain = args.pretrain)
    model = model.to(device)
    dataset = OriginDataset(args.data_root, args.stats_root, args.data_type, args.img_size, args.padding_size)
    mean, std = dataset.rgb_mean.astype(np.float32), dataset.rgb_std.astype(np.float32)
    if args.train_type == 'adv' or 'no' in args.pretrain:
        mean, std = np.zeros(mean.shape).astype(np.float32), np.ones(std.shape).astype(np.float32)
    tmean = torch.from_numpy(mean).to(device)
    tstd = torch.from_numpy(std).to(device)
    acc = 0
    for batch_id, (image, label, fname, category) in enumerate(dataset):

        perturb_image_root = os.path.join(args.perturb_root, category, fname)

        img = image[np.newaxis,:]
        preimg = (img - mean) / std
        timg = torch.from_numpy(preimg).to(device).float()
        out = model(timg).data.cpu().numpy()
        pred = np.argmax(out)
        print(f'{category} {fname} init pred: {pred} label: {label}')
        if pred != label:
            continue
        adv, tgt = cw_attack(device=device, img=image, label=label, model=model,
                        target=args.tlab, mean=tmean, std=tstd, confidence=0, max_iterations=1000, learning_rate=5e-3, initial_const=1e-2, binary_search_steps=5)
        if adv is None:
            print ('attack fail')
            continue
        mk_dir(perturb_image_root)
        perturb_image = adv - image
        to_save = {'ptb':perturb_image, 'tgt':tgt}
        np.save(f'{perturb_image_root}/ptb.npy', to_save)
