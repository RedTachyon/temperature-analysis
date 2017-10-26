import numpy as np

from utils import rolling_window


def mean_filter(data, window):
    pad = window // 2

    rolled = rolling_window(data, window).mean(1)
    rolled = np.hstack((np.ones((pad,)) * rolled[:pad].mean(), rolled, np.ones((pad,)) * rolled[-pad:].mean()))

    return rolled