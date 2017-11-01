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
    # X, Y1, Y2 = utils.read_data("data/data.pickle")
    #
    # data = dataholders.TempData(X, Y1)
    # data.preprocess(filters.median_filter, 15)
    #
    # data.process(heuristics.baseline_heuristic, 5)
    # data.detect(.015)
    #
    # # plt.plot(data.filtered)
    # # plt.plot(data.temperature - 14.2)
    # # plt.show()
    # print(data.time.shape)

    temp_path = 'data/raw/uft_flight07.mat'
    wind_path = 'data/raw/actos_flight07.mat'
    holder = dataholders.TempWindData(temp_path, wind_path)
    holder.cut_time(2750., 3000.)

    print(holder.v1.shape)

