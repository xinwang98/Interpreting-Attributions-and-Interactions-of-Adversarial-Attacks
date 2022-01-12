import numpy as np
import os
import csv

def cluster(src_path, tgt, patch_num, component_size, patch_threshold, global_threshold):
    val_id1 = np.genfromtxt(os.path.join(src_path, f'backshap{tgt//component_size}.csv'), delimiter=',')
    val_id2 = np.genfromtxt(os.path.join(src_path, f'backshap{tgt}.csv'), delimiter=',')
    val1, id1 = val_id1[:,0], val_id1[:,1:].astype(int)
    val2, id2 = val_id2[:, 0], val_id2[:, 1:].astype(int)
    res = []
    if tgt == component_size:    # combinations number
        combs = val2.size // (patch_num**2)
        selected_num = int(combs*patch_threshold)
        cnts, threshold = np.zeros(patch_num**2), [[] for _ in range(patch_num**2)]
        patch_diff = np.zeros((patch_num ** 2, combs))
        for idx in range(patch_num ** 2):
            for i in range(combs):
                gids = id2[combs * idx + i]
                pos = []
                for t in gids:
                    pos.append(np.argwhere(id1 == t)[0][0])
                patch_diff[idx][i] = np.abs(val2[combs * idx + i] - val1[pos].sum())
            sort = patch_diff[idx].argsort()[::-1]
            selected_ids = []
            for i in range(selected_num):
                flag = 0
                ids = id2[combs * idx+sort[i]]
                for j in ids:
                    if j in selected_ids:
                        flag = 1
                if flag == 0:
                    save = [val2[combs * idx + sort[i]]]
                    threshold[idx] = [i, patch_diff[idx][sort[i]]]
                    for j in ids:
                        selected_ids.append(j)
                        save.append(j)
                    res.append(save)
                    cnts[idx] += 1
            # print (f'patch {idx}: select {cnts[idx]}, threshold {threshold[idx]}')
        cnt = cnts.sum()

    else:
        diff = np.zeros(val2.size)
        # ------------------------- try different threshold -------------------------- #
        selected_num = int(val2.size * global_threshold)
        for i in range(val2.size):
            poss = []
            for t in range(0, tgt, tgt // component_size):
                poss.append(np.where(id1[:, 0] == id2[i][t])[0][0])
            poss = np.array(poss)
            diff[i] = np.abs(val2[i] - val1[poss].sum())

        sort = diff.argsort()[::-1] # sort[i]---pos in diff with large variance
        selected_ids = []
        cnt = 0
        for i in range(selected_num):
            flag = 0
            ids = id2[sort[i]]
            for j in ids:
                if j in selected_ids:
                    flag = 1
            if flag == 0:
                cnt += 1
                save = [val2[sort[i]]]
                threshold = [i, diff[sort[i]]]
                for j in ids:
                    selected_ids.append(j)
                    save.append(j)
                res.append(save)

        # print (f'finish cluster {tgt//4} to {tgt}, num change {val2.size} to {cnt}, threshold:{threshold}')
    with open(os.path.join(src_path, f'clusback{tgt}.csv'), 'w') as wf:
        wrt = csv.writer(wf)
        wrt.writerows(res)
