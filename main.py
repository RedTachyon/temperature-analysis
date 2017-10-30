import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import utils
import heuristics
import filters
import dataholders

plt.rcParams['figure.figsize'] = (18, 6)


if __name__ == "__main__":
    X, Y1, Y2 = utils.read_data("data/data.pickle")

    data = dataholders.TempData(X, Y1)
    data.preprocess(filters.median_filter, 15)

    data.process(heuristics.base_median_heuristic, 5)
    data.detect(.015)

    # plt.plot(data.filtered)
    # plt.plot(data.temperature - 14.2)
    # plt.show()
    data.plot()
