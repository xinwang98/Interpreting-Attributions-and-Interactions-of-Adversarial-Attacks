import torch
from codes.interaction import *
import numpy as np
import copy
import time
import random

GROUPS = 2
def generate_player(patch_size, super_size, patch_num, row, col):  # 2x2
    id_player = {}
    row_start = row * patch_size
    col_start = col * patch_size
    super_patch = patch_size // super_size
    head_id = patch_num * row * super_patch ** 2 + col * super_patch
    num_in_global = super_patch * patch_num
    for i in range(super_patch):
        for j in range(super_patch):
            idx = head_id + i*num_in_global + j
            player = (slice(0, 3),
                 slice(row_start + super_size*i, row_start + super_size*(i+1)),
                 slice(col_start + super_size*j, col_start + super_size*(j+1)))
            id_player[idx] = player
    return id_player

def to_global_id(row, col, patch_size, super_size, patch_num, ids):
    res = []
    super_patch = patch_size // super_size
    head_id = patch_num * row * super_patch ** 2 + col * super_patch
    for id in ids:
        r, c = id//super_patch, id%super_patch
        gid = head_id + r*super_patch*patch_num+c
        res.append(gid)
    return res

def generate_pairs(patch_num, patch_size, super_size, row, col, turn):
    players, ids = [], []
    row_start = row * patch_size
    col_start = col * patch_size
    super_patch = patch_size // super_size

    if turn == 0:   # (0,0)+(1,0) = 0+112
        for i in range(0, super_patch-1, 2):
            for j in range(super_patch):
                lids = [super_patch*i+j, super_patch*(i+1)+j]
                gids = to_global_id(row, col, patch_size, super_size, patch_num, lids)
                ids.append(gids)
                players.append(
                    (slice(0, 3),
                     slice(row_start + super_size * i, row_start + super_size * (i + 2)),
                     slice(col_start + super_size * j, col_start + super_size * (j + 1)))
                )  # channel, row, col
    elif turn == 1:   # (1,0)+(2,0) = 112+224
        for i in range(1, super_patch-1, 2):
            for j in range(super_patch):
                lids = [super_patch * i + j, super_patch * (i + 1) + j]
                gids = to_global_id(row, col, patch_size, super_size, patch_num, lids)
                ids.append(gids)
                players.append(
                    (slice(0, 3),
                     slice(row_start + super_size * i, row_start + super_size * (i + 2)),
                     slice(col_start + super_size * j, col_start + super_size * (j + 1)))
                )  # channel, row, col
    elif turn == 2:   # (0,0)+(0,1) = 0+1
        for i in range(super_patch):
            for j in range(0, super_patch-1, 2):
                lids = [super_patch * i + j, super_patch * i + 1 + j]
                gids = to_global_id(row, col, patch_size, super_size, patch_num, lids)
                ids.append(gids)
                players.append(
                    (slice(0, 3),
                     slice(row_start + super_size * i, row_start + super_size * (i + 1)),
                     slice(col_start + super_size * j, col_start + super_size * (j + 2)))
                )  # channel, row, col
    elif turn == 3:   # (0,1)+(0,2) = 1+2
        for i in range(super_patch):
            for j in range(1, super_patch-1, 2):
                lids = [super_patch * i + j, super_patch * i + 1 + j]
                gids = to_global_id(row, col, patch_size, super_size, patch_num, lids)
                ids.append(gids)
                players.append(
                    (slice(0, 3),
                     slice(row_start + super_size * i, row_start + super_size * (i + 1)),
                     slice(col_start + super_size * j, col_start + super_size * (j + 2)))
                )  # channel, row, col
    return players, ids

def generate_candidates(patch_size, super_size, patch_num, row, col):
    id_player = generate_player(patch_size, super_size, patch_num, row, col)
    final_players = []
    final_ids = []
    for turn in range(4):
        players, ids = generate_pairs(patch_num, patch_size, super_size, row, col, turn)

        random.shuffle(ids)
        for i in range(GROUPS):
            selected_ids = []
            start = i*(len(ids)//GROUPS)
            end = min(len(ids), (i+1)*(len(ids)//GROUPS))
            turn_idx = ids[start:end].copy()
            for pair in turn_idx:
                for p in pair:
                    selected_ids.append(p)
            turn_players = players[start:end].copy()
            for idx in id_player.keys():
                if idx in selected_ids:
                    continue
                turn_players.append(id_player[idx])
                turn_idx.append(np.array([idx]))
            final_players.append(turn_players)
            final_ids.append(turn_idx)

    return final_players, final_ids

def generate_quadruple(patch_num, patch_size, super_size, row, col, hr, hc, id_player):
    row_start, col_start = row * patch_size, col * patch_size
    super_patch = patch_size // super_size
    head_id = patch_num * row * super_patch ** 2 + col * super_patch
    tail_id = head_id + (super_patch-1)*super_patch*patch_num
    players = []
    unfixed = []
    if hr == 3:
        for t in range(head_id, head_id + super_patch):
            unfixed.append(id_player[t])
    if hr == 1:
        for t in range(tail_id, tail_id+super_patch):
            unfixed.append(id_player[t])
    if hc == 3:
        for t in range(head_id, tail_id+1, super_patch*patch_num):
            unfixed.append(id_player[t])
    if hc == 1:
        for t in range(head_id+super_patch-1, tail_id+super_patch, super_patch*patch_num):
            unfixed.append(id_player[t])
    for i in range(hr, super_patch-1, 4):
        for j in range(hc, super_patch-1, 4):
            belong = (row, col, super_patch*i+j, super_patch*i+j+1, super_patch*(i+1)+j, super_patch*(1+i)+j+1)
            idx = to_global_id(row, col, patch_size, super_size, patch_num, belong[2:])
            location = [
                (slice(0, 3),
                 slice(row_start + super_size * i, row_start + super_size * (i + 2)),
                 slice(col_start + super_size * j, col_start + super_size * (j + 2)))
            ] # channel, row, col
            player = ourPlayer(location, idx, belong)
            players.append(player)
    return players, unfixed


def generate_quacandidates(patch_size, super_size, patch_num, row, col, id_player):
    final_players, final_ids = [], []
    heads = []
    for r in range(4):
        for c in range(4):
            heads.append((r, c))
    for hr, hc in heads:
        players, unfixed = generate_quadruple(patch_num, patch_size, super_size, row, col, hr, hc, id_player)
        # --------------- ids can be chosen ---------------- #
        for i in range(GROUPS):
            single_players = unfixed.copy()
            start = i * (len(players) // GROUPS)
            end = (i+1) * (len(players) // GROUPS)
            if i == GROUPS-1:
                end = max(len(players), (i + 1) * (len(players) // GROUPS))
            turn_players = players[start:end].copy()
            for quad in players:
                if quad not in turn_players:
                    for p in quad.idx:
                        single_players.append(id_player[p])

            final_players.append(turn_players+single_players)

    return final_players

if __name__ == '__main__':
    id_player = generate_single_player(28, 2, 8)
    res = generate_quacandidates(28,2,8,1,0,id_player)
    print (len(res))
    print (len(res[1]))
    for p in res[1]:
        print (p.idx)