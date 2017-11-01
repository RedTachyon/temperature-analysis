import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


import utils
import heuristics
import filters


class TempData:
    """
    Holds the data about a single temperature time series.
    """

    def __init__(self, time, temperature):
        """

        Args:
            time: array, holding the time indices
            temperature: array, holding the temperature readings
        """
        self.time = time
        self.temperature = temperature

        self.baseline = None
        self.processed = None
        self.detected = None
        self.filtered = None
        self.mask = None

    def preprocess(self, method=lambda x: x, *args, **kwargs):
        """
        Filters the temperature with a chosen function.

        Args:
            method: function, filter to be applied to the temperature

        Returns:
            filtered_temperature, array
        """
        self.baseline = method(self.temperature, *args, **kwargs)
        self.filtered = self.temperature - self.baseline
        # return self.baseline

    def process(self, method, window):
        """
        Converts the temperature data to values proportional to the likelihood of being jumps.

        Args:
            method: function array => array, algorithm to be used
            window: int, size of the rolling window

        Returns:
            array
        """

        if self.baseline is None:
            raise AttributeError("You must run preprocessing first. "
                                 "If you don't know what to use, just use an identity filter")

        assert window % 2

        pad = window // 2

        strided_data = utils.rolling_window(self.filtered, window)

        self.processed = np.hstack((np.zeros(pad), method(strided_data), np.zeros(pad)))
        # return self.processed

    def detect(self, threshold):
        """
        Sets the values above threshold to True, and the rest to False.

        Args:
            threshold: float

        Returns:
            boolean array
        """
        if self.processed is None:
            raise AttributeError("You must run processing first. "
                                 "If you don't know what to use, just use a simple max - min filter")

        self.mask = self.processed > threshold
        detected = np.where(self.mask)[0] / 100.
        detected += self.time[0]

        self.detected = detected

        # return self.detected

    def plot(self, base=False):
        plt.plot(self.time, self.temperature, linewidth=.5, color='b')
        if base:
            plt.plot(self.time, self.baseline, linewidth=1.2, color='y')
        if self.detected is not None:
            plt.vlines(self.detected, 13.8, 14.6, alpha=.25, linewidth=2.5, colors='r')
        plt.show()

    def plot_base(self):
        plt.plot(self.time, self.temperature, linewidth=1.2, color='r')
        plt.plot(self.time, self.baseline, linewidth=1.2)
        plt.show()


class WindData:
    """Holds the information about wind-related data."""

    def __init__(self, wind_x, wind_y, wind_z, degrees=False):
        self.wind = (wind_x, wind_y, wind_z)
        self.theta, self.phi = None, None

        self._get_angles(degrees=degrees)

    def _get_angles(self, degrees=False):
        self.theta = np.arccos(self.wind[2] / np.sqrt(self.wind[0] ** 2 + self.wind[1] ** 2 + self.wind[2] ** 2))
        self.theta = self.theta.ravel()

        self.phi = np.arccos(self.wind[1] / np.sqrt(self.wind[0] ** 2 + self.wind[1] ** 2))
        self.phi = self.phi.ravel()

        if degrees:
            self.theta *= 180 / np.pi
            self.phi *= 180 / np.pi


class TempWindData:
    """
    Holds the temperature and wind data from a single flight.
    """

    def __init__(self, path_temp, path_wind):
        self.path_temp = path_temp
        self.path_wind = path_wind

        self.v1, self.v2, self.v3, self.X_temp, self.X_wind, self.T1, self.T2 = None, None, None, None, None, None, None
        self.time = None

        self._load_data()
        self._synchronize()

    def _load_data(self):
        """Loads the temperature and wind data from .mat files."""
        winddata = loadmat(self.path_wind, variable_names=('sonic1', 'sonic2', 'sonic3',))
        tempdata = loadmat(self.path_temp, variable_names=('lowT_av', 'upT_av', 'time_av'))

        self.v1, self.v2, self.v3 = winddata['sonic1'].ravel(), winddata['sonic2'].ravel(), winddata['sonic3'].ravel()
        self.X_wind = np.arange(self.v1.shape[0]) / 100.

        self.X_temp = tempdata['time_av'].ravel()
        self.T1 = tempdata['lowT_av'].ravel()
        self.T2 = tempdata['upT_av'].ravel()

    def _synchronize(self):
        """Synchronizes the wind and temperature data."""
        low = np.max((self.X_temp[0], self.X_wind[0]))
        self.X_temp -= (self.X_temp[self.X_temp >= low][0] - self.X_wind[self.X_wind >= low][0])

        high = np.min((self.X_temp[-1], self.X_wind[-1])) + 1e-9  # Includes a small constant to keep the same shape

        self.v1 = utils.array_range(self.v1, low, high, self.X_wind)
        self.v2 = utils.array_range(self.v2, low, high, self.X_wind)
        self.v3 = utils.array_range(self.v3, low, high, self.X_wind)
        self.X_wind = utils.array_range(self.X_wind, low, high, self.X_wind)

        self.T1 = utils.array_range(self.T1, low, high, self.X_temp)
        self.T2 = utils.array_range(self.T2, low, high, self.X_temp)
        self.X_temp = utils.array_range(self.X_temp, low, high, self.X_temp)

        self.time = self.X_wind

    def cut_time(self, low, high):
        """Restricts the data to a specific time."""
        self.v1 = utils.array_range(self.v1, low, high, self.X_wind)
        self.v2 = utils.array_range(self.v2, low, high, self.X_wind)
        self.v3 = utils.array_range(self.v3, low, high, self.X_wind)
        self.X_wind = utils.array_range(self.X_wind, low, high, self.X_wind)

        self.T1 = utils.array_range(self.T1, low, high, self.X_temp)
        self.T2 = utils.array_range(self.T2, low, high, self.X_temp)
        self.X_temp = utils.array_range(self.X_temp, low, high, self.X_temp)

        self.time = utils.array_range(self.time, low, high, self.time)


if __name__ == '__main__':
    temp_path = 'data/raw/uft_flight07.mat'
    wind_path = 'data/raw/actos_flight07.mat'

    holder = TempWindData(temp_path, wind_path)

    holder.cut_time(2750, 3000)

    print(holder.v1.shape)
