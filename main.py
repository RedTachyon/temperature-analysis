import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import utils
import heuristics
import filters

plt.rcParams['figure.figsize'] = (18, 6)


class Series:
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

        self.filtered_temperature = None
        self.processed = None
        self.detected = None

    def preprocess(self, method=lambda x: x, *args, **kwargs):
        """
        Filters the temperature with a chosen function.

        Args:
            method: function array => array, filter to be applied to the temperature

        Returns:
            filtered_temperature, array
        """
        self.filtered_temperature = method(self.temperature, *args, **kwargs)
        return self.filtered_temperature

    def process(self, method, window):
        """
        Converts the temperature data to values proportional to the likelihood of being jumps.

        Args:
            method: function array => array, algorithm to be used
            window: int, size of the rolling window

        Returns:
            array
        """

        if self.filtered_temperature is None:
            raise AttributeError("You must run preprocessing first. "
                                 "If you don't know what to use, just use an identity filter")

        assert window % 2

        pad = window // 2

        strided_data = utils.rolling_window(self.filtered_temperature, window)

        self.processed = np.hstack((np.zeros(pad), method(strided_data), np.zeros(pad)))
        return self.processed

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

        return self.detected

    def plot(self, **kwargs):
        plt.plot(self.time, self.temperature, **kwargs)
        if self.detected is not None:
            plt.vlines(self.detected, 13.8, 14.6, alpha=.25, linewidth=2.5, colors='r')
        plt.show()

if __name__ == "__main__":
    X, Y1, Y2 = utils.read_data("data.pickle")

    data = Series(X, Y1)
    data.preprocess(filters.mean_filter, 21)

    plt.plot(data.temperature)
    plt.plot(data.filtered_temperature)
    plt.show()

    # data.process(heuristics.base_mean_heuristic, 5)
    # data.detect(.015)
    #
    # data.plot(linewidth=.5)
