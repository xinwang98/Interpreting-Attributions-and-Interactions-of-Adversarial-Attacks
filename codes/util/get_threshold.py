import matplotlib.pyplot as plt
import numpy as np


def get_threshold(arr, rate=0.05):
    values, bins, _ = plt.hist(arr.flatten(), bins=100, density=True, cumulative=True)
    left = np.argmin(np.abs(values - rate))
    right = np.argmin(np.abs(values - (1 - rate)))
    return bins[left], bins[right]