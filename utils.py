import numpy as np
import pickle


def rolling_window(series, window):
    """
    Helper function for performing rolling window computations.
    Adds an extra dimension to the array, which can then be used to do whatever you need to do on windows.

    Args:
        series: np.array, series to be unrolled
        window: int, size of the rolling window

    Returns:
        np.array
    """
    shape = series.shape[:-1] + (series.shape[-1] - window + 1, window)
    strides = series.strides + (series.strides[-1],)
    return np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)


def read_pickle(path):
    """Reads an array from the path, using pickle"""
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def read_data(path, low=275000, high=300000):
    """Reads data and preformats it."""
    data = read_pickle(path)

    Y1 = data['lowT_av'].squeeze()
    Y2 = data['upT_av'].squeeze()
    # LWC = data['lwc1V_av']
    X = np.arange(Y1.shape[0]) / 100.
    # X = data['time_av'].squeeze()
    if low is not None and high is not None:
        X = X[low:high]
        Y1 = Y1[low:high]
        Y2 = Y2[low:high]

    return X, Y1, Y2


def array_range(a, low, high, ref=None):
    """
    Returns the array limited to values in selected range.
    """
    if ref is None:
        ref = a
    return a[np.logical_and(ref >= low, ref < high)]