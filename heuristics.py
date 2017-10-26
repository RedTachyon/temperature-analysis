import numpy as np


def baseline_heuristic(data):
    # Decent threshold: 0.01
    return data.max(axis=1) - data.min(axis=1)


def base_mean_heuristic(data):
    # Decent threshold: 0.007
    return data.max(axis=1) - data.mean(axis=1)


def base_median_heuristic(data):
    return data.max(axis=1) - np.median(data, axis=1)


def std_small_heuristic(data):
    # Decent threshold: 5
    return (data.max(axis=1) - data.min(axis=1)) / data.std(axis=1)


def std_wide_heuristic(data):
    N = data.shape[1]
    mid = int((N - 1) / 2)
    data = data / data.std(axis=1, keepdims=True)
    return data[:, mid - 2:mid + 3].max(axis=1) - data[:, mid - 2:mid + 3].min(axis=1)


def std_new_heuristic(data):
    N = data.shape[1]
    mid = (N - 1) / 2
    data = data / data.std(axis=1, keepdims=True)
    return data[:, mid:N + 1] - data[:, N - 2:N + 3].min(axis=1)

