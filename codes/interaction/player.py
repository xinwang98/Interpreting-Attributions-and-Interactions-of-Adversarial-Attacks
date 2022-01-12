from itertools import permutations

import numpy as np


class ourPlayer(object):
    def __init__(self, location, idx=[-1], belong=(-1,-1,-1), img_size=112):
        self.location = location
        self.idx = tuple(idx)
        self.belong = np.array(belong)
        rows, cols = 0, 0
        for k in self.idx:
            rows += int(k) // img_size
            cols += int(k) % img_size
        self.center = np.array([rows/len(self.idx), cols/len(self.idx)])

    def __len__(self):
        return len(self.location)

    # def __add__(self, other):
    #     return Player(
    #         self.location + other.location,
    #         self.value + other.value,
    #         list(self.idx) + list(other.idx)
    #     )

def generate_single_player(row=-1, col=-1, patch_size=28, super_size=4, patch_num=8):  # 2x2
    super_patch = patch_size // super_size
    num_in_global = super_patch * patch_num
    pixels = list(permutations(range(super_patch),2))
    patches = list(permutations(range(patch_num), 2))
    players = {}
    for t in range(patch_num):
        patches.append((t, t))
    for t in range(super_patch):
        pixels.append((t,t))
    if row==-1 and col==-1:
        for (row,col) in patches:
            row_start = row * patch_size
            col_start = col * patch_size
            head_id = patch_num * row * super_patch ** 2 + col * super_patch
            for (i,j) in pixels:
                idx = head_id + i*num_in_global + j
                location = [(slice(0, 3),
                     slice(row_start + super_size*i, row_start + super_size*(i+1)),
                     slice(col_start + super_size*j, col_start + super_size*(j+1)))]
                belong = (row, col, i*super_patch+j)
                players[idx] = ourPlayer(location,[idx], belong)
    else:
        row_start = row * patch_size
        col_start = col * patch_size
        head_id = patch_num * row * super_patch ** 2 + col * super_patch
        for (i, j) in pixels:
            idx = head_id + i * num_in_global + j
            location = [(slice(0, 3),
                        slice(row_start + super_size * i, row_start + super_size * (i + 1)),
                        slice(col_start + super_size * j, col_start + super_size * (j + 1)))]
            belong = (row, col, i * super_patch + j)
            players[idx] = ourPlayer(location, [idx], belong)
    return players

