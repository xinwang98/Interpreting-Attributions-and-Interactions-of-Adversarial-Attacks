from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import PIL
import os
import sys
import numpy as np


def get_label(path):
    label_name = []
    f = open(path, 'r')
    for line in f.readlines():
        label_name.append(line[:-1])
    f.close()
    label_idx = {label_name[i]: i for i in range(len(label_name))}
    return label_name, label_idx


def get_stats(stats_root, padding_size):
    if stats_root and padding_size:
        rgb_mean = np.load(f'{stats_root}/rgb_mean.npy').reshape((3, 1, 1))
        rgb_std = np.load(f'{stats_root}/rgb_std.npy').reshape((3, 1, 1))
    else:
        rgb_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        rgb_std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    return rgb_mean, rgb_std

class OriginDataset(Dataset):
    def __init__(self, data_root, stats_root, data_type, img_size, padding_size):
        rgb_mean, rgb_std = get_stats(stats_root, padding_size)  # 0-1
        self.rgb_mean = rgb_mean.reshape((3, 1, 1))
        self.rgb_std = rgb_std.reshape((3, 1, 1))
        label_name, label_idx = get_label(f'{stats_root}/{data_type}_list.txt')
        self.label_name, self.label_idx = label_name, label_idx
        inner_size = img_size - 2 * padding_size
        self.transform = transforms.Compose([
            transforms.Resize((inner_size, inner_size)),
            transforms.Pad(padding_size,
                           fill=(127, 127, 127))  # padding with grey
        ])

        samples = self.get_samples(path=data_root)
        self.samples = samples

    def get_samples(self, path):
        samples = []
        for category in sorted(os.listdir(path)):
            cate_path = os.path.join(path, category)
            for img_name in sorted(os.listdir(cate_path)):
                img_path = os.path.join(cate_path, img_name)
                label = self.label_idx[category]
                item = (img_path, category, label, img_name)
                samples.append(item)
        return samples

    def __getitem__(self, index):  # image(numpy, 3*224*224, 0-1)
        path, category, label, fname = self.samples[index]
        img = Image.open(path).convert('RGB')
        img = np.array(self.transform(img)) / 255
        img = np.transpose(img, (2, 0, 1))
        return img, label, fname, category

    def __len__(self):
        return len(self.samples)
