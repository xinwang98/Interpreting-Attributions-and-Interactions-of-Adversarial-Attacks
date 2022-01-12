import copy

import numpy as np
import torch
from codes.util import *


def backward_shap(model, label, image, perturbation, players, sample_num, rgb_mean, rgb_std, target, seed, device, subplayer_num=10):
    for param in model.parameters():
        param.requires_grad = False
    np.random.seed(seed)
    player_num = len(players)
    image = torch.from_numpy(image[np.newaxis, ...]).float().to(device)
    rgb_mean = torch.from_numpy(rgb_mean).float().to(device)
    rgb_std = torch.from_numpy(rgb_std).float().to(device)

    subplayer_value = perturbation / subplayer_num
    grad_values = np.zeros_like(perturbation)
    for _ in range(sample_num):
        order = np.arange(player_num).repeat(subplayer_num)
        subplayer_order = np.random.permutation(order)
        block_perturbation = copy.deepcopy(perturbation.astype(np.float32))

        for i in range(player_num * subplayer_num):
            player = players[subplayer_order[i]]
            for pos in player.location:
                block_perturbation[pos] -= subplayer_value[pos]

            tensor_block_perturbation = torch.from_numpy(block_perturbation).to(device).requires_grad_(True)
            model_input = (image + tensor_block_perturbation - rgb_mean) / rgb_std
            y = model(model_input)
            v = y[0,target] - y[0,label]
            v.backward()
            grad = tensor_block_perturbation.grad.detach().cpu().numpy()
            grad_values += grad

    shap_dict = {}
    shap_map = grad_values * subplayer_value / (sample_num * player_num)
    for player in players:
        shap = 0
        for pos in player.location:
            shap += shap_map[pos].sum()
        shap_dict[player.idx] = shap
    return shap_dict


def backward_shap_new(model,
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
                      subplayer_num=10,
                      batch_size=32):
    for param in model.parameters():
        param.requires_grad = False
    np.random.seed(seed)
    player_num = len(players)
    image = torch.from_numpy(image[np.newaxis, ...]).float().to(device)
    rgb_mean = torch.from_numpy(rgb_mean).float().to(device).reshape(
        (1, 3, 1, 1))
    rgb_std = torch.from_numpy(rgb_std).float().to(device).reshape(
        (1, 3, 1, 1))

    subplayer_value = torch.from_numpy(
        perturbation / subplayer_num).unsqueeze(0).to(device).float()
    grad_values = np.zeros_like(perturbation)
    for _ in range(sample_num):
        order = np.arange(player_num).repeat(subplayer_num)
        subplayer_order = np.random.permutation(order)
        block_perturbation = copy.deepcopy(perturbation.astype(np.float32))
        block_perturbation = torch.from_numpy(block_perturbation).to(
            device).repeat(len(order), 1, 1, 1)

        for i in range(player_num * subplayer_num):
            player = players[subplayer_order[i]]
            for pos in player.location:
                block_perturbation[i:, pos[0], pos[1],
                                   pos[2]] -= subplayer_value[:, pos[0],
                                                              pos[1], pos[2]]

            # tensor_block_perturbation = torch.from_numpy(
            #     block_perturbation).to(device).requires_grad_(True)
        start = 0
        end = start + batch_size
        while start < len(order):
            tensor_block_perturbation = block_perturbation[start:end, :, :, :]
            tensor_block_perturbation.requires_grad = True
            model_input = (image + tensor_block_perturbation -
                           rgb_mean) / rgb_std
            y = model(model_input)
            v = (y[:, target] - y[:, label]).sum()
            v.backward()
            grad = tensor_block_perturbation.grad.detach().cpu().numpy().sum(0)
            grad_values += grad
            start = end
            end = min(len(order), end + batch_size)

    subplayer_value = subplayer_value.cpu().numpy()
    shap_dict = {}
    shap_map = grad_values * subplayer_value / (sample_num * player_num)
    shap_map = shap_map.squeeze()
    for player in players:
        shap = 0
        for pos in player.location:
            shap += shap_map[pos].sum()
        shap_dict[player.idx] = shap
    return shap_dict


@timeit
def gbackward_shap(model, label, image, perturbation, players, sample_num, rgb_mean, rgb_std, target, seed, device, subplayer_num=10):
    for param in model.parameters():
        param.requires_grad = False
    np.random.seed(seed)
    player_num = len(players)
    image = torch.from_numpy(image[np.newaxis, ...]).float().to(device)
    rgb_mean = torch.from_numpy(rgb_mean).float().to(device)
    rgb_std = torch.from_numpy(rgb_std).float().to(device)

    subplayer_value = perturbation / subplayer_num
    grad_values = np.zeros_like(perturbation)
    for _ in range(sample_num):
        order = np.arange(player_num).repeat(subplayer_num)
        subplayer_order = np.random.permutation(order)

        block_perturbation = copy.deepcopy(perturbation.astype(np.float32))

        for i in range(player_num * subplayer_num):
            player = players[subplayer_order[i]]
            # --------------------- generate masked perturbation ----------------------- #
            for i in range(len(player)): # a player contain several irregualar pixels
                block_perturbation[player[i]] -= subplayer_value[player[i]]

            tensor_block_perturbation = torch.from_numpy(block_perturbation).to(device).requires_grad_(True)
            model_input = (image + tensor_block_perturbation - rgb_mean) / rgb_std
            y = model(model_input)
            # y[0, target].backward()
            v = y[0,target] - y[0,label]
            v.backward()

            grad = tensor_block_perturbation.grad.detach().cpu().numpy()
            grad_values += grad

    shap_value = np.zeros(player_num)
    shap_map = grad_values * subplayer_value / (sample_num * player_num)
    for i in range(player_num):
        player = players[i]
        for j in range(len(player)):
            shap_value[i] += shap_map[player[j]].sum()
    return shap_value
