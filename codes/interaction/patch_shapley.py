import torch
import numpy as np
import copy
from codes.interaction import generate_single_player, backward_shap, generate_quacandidates, backward_shap_new
from codes.util import *
import time

def patch_shap(model, label, image, perturbation, rgb_mean, rgb_std, target, device, seed,
               shapley_method, batch_size, component_size, super_size=4, patch_size=28, patch_num=8, sample_num=1):
    shap_save = []
    for row in range(patch_num):
        for col in range(patch_num):
            if shapley_method[-1] == '1':
                id_player = generate_single_player(row, col, patch_size, super_size, patch_num)
                shap_dict = backward_shap_new(
                    model,
                    label,
                    image,
                    perturbation,
                    list(id_player.values()),
                    sample_num,
                    rgb_mean,
                    rgb_std,
                    target,
                    seed,
                    device,
                    batch_size=batch_size)
                for k in shap_dict.keys():
                    res = [shap_dict[k]]
                    for i in k:
                        res.append(i)
                    shap_save.append(res)

            if shapley_method[-1] == str(component_size):
                group_info = []
                id_player = generate_single_player(row, col, patch_size, super_size, patch_num)
                all_players = generate_quacandidates(patch_size, super_size, patch_num, row, col, id_player)
                for i, players in enumerate(all_players):
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
                        if len(k) < component_size:
                            continue
                        res = [shap_dict[k]]
                        for i in k:
                            res.append(i)
                        group_info.append(res)
                for info in group_info:
                    shap_save.append(info)

    return np.array(shap_save)
