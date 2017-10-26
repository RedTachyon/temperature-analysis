import numpy as np
import pickle


def rolling_window(series, window):
    """
    Helper function for performing rolling window computations.
    Adds an extra dimension to the array, which can then be used to do whatever you need to do on windows.

    Args:
        series: array, series to be unrolled
        window: int, size of the rolling window

    Returns:
        array
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

    X_cut = X[low:high]
    Y1_cut = Y1[low:high]
    Y2_cut = Y2[low:high]

    return X_cut, Y1_cut, Y2_cut
