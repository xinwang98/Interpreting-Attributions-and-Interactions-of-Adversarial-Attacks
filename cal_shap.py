import csv
import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataset import get_label
from codes.util import *

def shapley(model_type, data_type, norm='l2', tgt=0):
    total_root = f'./experiment/attribution_{tgt}/CSV/{norm}/{model_type}/{data_type}'

    for cate in os.listdir(total_root):
        for img in os.listdir(f'{total_root}/{cate}'):
            root = f'{total_root}/{cate}/{img}'

            print (f'cumpute: {cate} {img}')
            avail = [i for i in range(65)]
            write = os.path.join(root, 'shap.csv')
            shap = []
            with open(write, 'w') as f:
                writer = csv.writer(f)
                masks = [i for i in range(64)]
                writer.writerow(masks)
                for num in avail:
                    ex_sum, in_sum = [0.0] * 64, [0.0] * 64
                    ex_cnt, in_cnt = [0.0] * 64, [0.0] * 64
                    ex_mean, in_mean = [], []
                    path = os.path.join(root, str(num)+'.csv')
                    f = open(path, "r")
                    reader = csv.reader(f)
                    for line in reader:
                        if line[-1] == '0':
                            continue
                        for idx in range(64):
                            if str(idx) in line[:-2]:
                                in_cnt[idx] += 1
                                in_sum[idx] += (float(line[-2]))
                            else:
                                ex_cnt[idx] += 1
                                ex_sum[idx] += (float(line[-2]))
                    for i in range(64):
                        if ex_cnt[i]:
                            ex_mean.append(ex_sum[i]/ex_cnt[i])
                        else:
                            ex_mean.append(0.0)
                        if in_cnt[i]:
                            in_mean.append(in_sum[i]/in_cnt[i])
                        else:
                            in_mean.append(0.0)
                    shap.append(ex_mean)
                    shap.append(in_mean)
                    writer.writerows([ex_mean, in_mean])
                shap = np.array(shap)
                val = [0.0] * 64
                for i in range(64):
                    for j in range(64):
                        val[i] += float(shap[2 * j + 3][i])
                        val[i] -= float(shap[2 * j][i])
                    val[i] /= 64
                writer.writerow(val)

if __name__ == '__main__':
    for model in ['res18']:
        for data in ['cub']: #'voc', 'cub', 'sdd']:
            for tgt in [35]:
                print (model, data, tgt)
                shapley(model, data, norm='l2', tgt=tgt)
            # draw(model, data)