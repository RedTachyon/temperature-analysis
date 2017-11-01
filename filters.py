import numpy as np
from scipy.signal import butter, lfilter

from utils import rolling_window


def mean_filter(data, window):
    pad = window // 2

    rolled = rolling_window(data, window).mean(1)
    rolled = np.hstack((np.ones((pad,)) * rolled[:pad].mean(), rolled, np.ones((pad,)) * rolled[-pad:].mean()))

    return rolled


def median_filter(data, window):
    pad = window // 2

    rolled = np.median(rolling_window(data, window), 1)
    rolled = np.hstack((np.ones((pad,)) * rolled[:pad].mean(), rolled, np.ones((pad,)) * rolled[-pad:].mean()))

    return rolled


def butter_bandpass(lowcut, highcut, fs, order=5):
    """This is straight from the scipy cookbook"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
