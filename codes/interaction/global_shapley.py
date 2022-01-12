from codes.interaction import generate_single_player, search_quecandidates, candidates_to_players, backward_shap, backward_shap_new
import numpy as np
import time
import os

def global_shap(clus_path, model, label, image, perturbation, rgb_mean, rgb_std, target, device, seed, shapley_method, batch_size, img_size, patch_size, patch_num, super_size=4, sample_num=1, threshold=0.3, component_size=4):
    tgt = int(shapley_method[8:])
    # print(f'size {tgt} shapley value compute')
    pixel = generate_single_player(
        row=-1, col=-1, patch_size=patch_size, super_size=super_size, patch_num=patch_num)
    path = os.path.join(clus_path, f'clusback{tgt//component_size}.csv')
    val, pids = np.genfromtxt(path, delimiter=',')[:, 0], np.genfromtxt(path, delimiter=',')[:, 1:].astype(int)
    exist = []
    shap_save = []
    while (len(exist) < threshold * val.size):
        # start = time.time()
        group, exist = search_quecandidates(val, pids, img_size, super_size, candidates=exist)
        players = candidates_to_players(group, pixel)
        shap_dict = backward_shap_new(
            model,
            label,
            image,
            perturbation,
            players,
            sample_num,
            rgb_mean,
            rgb_std,
            target,
            seed,
            device,
            batch_size=batch_size)
        for k in shap_dict.keys():
            if len(k) < tgt:
                continue
            res = [shap_dict[k]]
            for i in k:
                res.append(i)
            shap_save.append(res)
        # end = time.time()
        # print (f'player number: {len(group)} overall coalition: {len(exist)} cost {(end-start):2f}')
    return np.array(shap_save)
