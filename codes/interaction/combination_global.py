import os
import numpy as np
import random
import copy
from codes.interaction import *

GROUPS = 2
# ------------------- pair --------------------- #
def judge_quadrant(src, tgt):
    r, c = tgt[0], tgt[1]
    r0, c0 = src[0], src[1]
    if (r<=r0+c0-c) and (r<c+r0-c0):
        return 0 # up
    if (r>r0+c0-c) and (r<=c+r0-c0):
        return 1 # right
    if (r>=r0+c0-c) and (r>c+r0-c0):
        return 2 # down
    if (r<r0+c0-c) and (r>=c+r0-c0):
        return 3 # left
    return -1

def search_candidates(clus_path, src, img_size, super_size, task='back'):
    path = os.path.join(clus_path, f'clus{task}{src}.csv')
    val, pids = np.genfromtxt(path, delimiter=',')[:,0], np.genfromtxt(path, delimiter=',')[:,1:].astype(int)
    # ------------------------- calculate distance in current state --------------------------- #
    cens = []
    super_img = img_size // super_size
    for ids in pids:
        rs, cs = ids//super_img, ids%super_img
        rmean, cmean = rs.sum()/ids.size, cs.sum()/ids.size
        cens.append([rmean, cmean])
    ncens = np.array(cens)
    dis = np.ones((val.size, val.size))*30000
    for i in range(val.size):
        for j in range(val.size):
            if i != j:
                dis[i][j] = np.sum((ncens[i]-ncens[j])**2) # L2
    # -------------------------- select four directions ----------------------------------------#
    res = []
    selected = []
    for i in range(val.size):
        flag = np.ones(4).astype(int)
        ids = np.argsort(dis[i])
        for id in ids:
            if flag.sum()==0:
                break
            else:
                pos = judge_quadrant(cens[i], cens[id])
                if pos>=0 and flag[pos]!=0:
                    flag[pos] = 0
                    tmp = [cens[id], cens[i]]
                    if tmp not in selected:
                        selected.append([cens[i], cens[id]])
                        res.append(np.hstack((pids[i], pids[id])))

    res = np.array(res)
    return res, pids

def generate_global_players(candidates, pids, exist_cdd=[]):
    players_ids = []
    prev = []
    for i, cdd in enumerate(candidates):
        flag = 0
        if i in exist_cdd:
            continue
        for cd in cdd:
            if cd in prev:
                flag = 1
        if flag == 0:
            exist_cdd.append(i)
            players_ids.append(cdd)
            for cd in cdd:
                prev.append(cd)
    final_players = []
    random.shuffle(players_ids)
    for i in range(GROUPS):
        start = i * (len(players_ids)//GROUPS)
        end = min(len(players_ids),(i+1) * (len(players_ids)//GROUPS))
        turn_ids = players_ids[start:end].copy()
        for p in pids:
            exist = 0
            for t in turn_ids:
                if p[0] in t:
                    exist = 1
            if not exist:
                turn_ids.append(p)
        final_players.append(turn_ids)
    final_players = np.array(final_players)
    return final_players, exist_cdd

def ids_to_player(players_ids, img_size, super_size):
    players = []
    super_img = img_size // super_size
    for player_ids in players_ids:
        player = []
        for id in player_ids:
            r, c = id//super_img, id%super_img
            player.append(
                (slice(0, 3),
                 slice(super_size*r, super_size*(r+1)),
                 slice(super_size*c, super_size*(c+1)))
            )  # channel, row, col
        players.append(player)
    return players

def search_quecandidates(val, pids, img_size, super_size, candidates=[]):
    # ------------------------- calculate distance in current state --------------------------- #
    cens = []
    super_img = img_size // super_size
    for ids in pids:
        rs, cs = ids // super_img, ids % super_img
        rmean, cmean = rs.sum() / ids.size, cs.sum() / ids.size
        cens.append([rmean, cmean])
    ncens = np.array(cens)
    dis = np.ones((val.size, val.size)) * 1e8
    for i in range(val.size):
        for j in range(val.size):
            if i != j:
                dis[i][j] = np.sum((ncens[i] - ncens[j]) ** 2)  # L2
    near_list, group = [], []
    no_select = set() # no_select save order_idx in csv
    for i in range(val.size):
        near_list.append(list(np.argsort(dis[i])))

    # ----------------- coalition player ------------------- #
    for i in range(val.size):
        # -------------- check player the current coalion (fixed or selected) --------------- #
        select = 1
        if i in no_select or i in candidates:
            select = 0
        for j in near_list[i][:3]:
            if j in no_select:
                select = 0
        if select != 1:
            continue
        # ----------------- form the coalition (pixel id) ------------------ #
        ids = list(pids[i].copy())
        for t in near_list[i][:3]:
            ids = ids + list(pids[t].copy())
        group.append(ids)
        candidates.append(i)
        # ------------------ throw out players (fixed or selected) ------------------ #
        for t in near_list[i][:3]:
            no_select = no_select | set([t])
            no_select = no_select | set(near_list[t][:3])
        no_select = no_select | set([i])
    # ------------------ single_player ------------------ #
    for i in range(val.size):
        if i not in candidates and i not in no_select:
            group.append(pids[i])

    return group, candidates

def candidates_to_players(group, pixel):
    players = []
    for ids in group:
        idx = ids
        location = []
        for t in idx:
            location = location + pixel[t].location
        players.append(ourPlayer(location, idx))
    return players





