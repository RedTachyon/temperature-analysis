import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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

        detected = np.where(self.processed > threshold)[0] / 100.
        detected += self.time[0]

        self.detected = detected

        # return self.detected

    def plot(self, **kwargs):
        plt.plot(self.time, self.temperature, linewidth=.5)
        if self.detected is not None:
            plt.vlines(self.detected, 13.8, 14.6, alpha=.25, linewidth=2.5, colors='r')
        plt.show()

    def plot_base(self):
        plt.plot(self.time, self.temperature, linewidth=1.2, color='r')
        plt.plot(self.time, self.baseline, linewidth=1.2)
        plt.show()


class WindData:
    """Holds the information about wind-related data."""

    def __init__(self, wind_x, wind_y, wind_z):
        self.wind = (wind_x, wind_y, wind_z)

    def _get_angles(self):
        self.theta = np.arccos(self.wind[2] / np.sqrt(self.wind[0] ** 2 + self.wind[1] ** 2 + self.wind[2] ** 2))
        self.theta = self.theta.ravel()

        self.phi = np.arccos(self.wind[1] / np.sqrt(self.wind[0] ** 2 + self.wind[1] ** 2))
        self.phi = self.phi.ravel()
