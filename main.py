#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np

import dataholders
import filters
import heuristics

plt.rcParams['figure.figsize'] = (18, 6)

DEGREES = True


def plot_histograms(holder):

    # Limit the data to the interesting interval
    holder.cut_time(2700., 3000.)

    # Extract and process the temperature data
    temp_data1 = dataholders.TempData(holder.time, holder.T1)
    temp_data1.preprocess(filters.median_filter, 21)
    temp_data1.process(heuristics.base_median_heuristic, 7)
    temp_data1.detect(.015)

    temp_data2 = dataholders.TempData(holder.time, holder.T2)
    temp_data2.preprocess(filters.median_filter, 21)
    temp_data2.process(heuristics.base_median_heuristic, 5)
    temp_data2.detect(.010)

    # Extract and process the wind data
    wind_data = dataholders.WindData(holder.v1, holder.v2, holder.v3, degrees=DEGREES)

    print("Percentage of anomalies in T1: %.2f%%" % (temp_data1.mask.mean() * 100))
    print("Percentage of anomalies in T2: %.2f%%" % (temp_data2.mask.mean() * 100))

    print(temp_data1.mask.sum())
    print(temp_data2.mask.sum())

    both_mask = np.logical_or(temp_data1.mask, temp_data2.mask)

    print(both_mask.sum())

    print("Percentage of anomalies in either: %.2f%%" % (both_mask.mean() * 100))

    # Select the angles with and without anomalies

    mask = temp_data1.mask

    theta_bad = wind_data.theta[mask]
    phi_bad = wind_data.phi[mask]

    theta_good = wind_data.theta[~mask]
    phi_good = wind_data.phi[~mask]

    # Compute the histograms
    hist_bad, xedg_bad, yedg_bad = np.histogram2d(theta_bad, phi_bad, bins=100, normed=True)
    hist_good, xedg_good, yedg_good = np.histogram2d(theta_good, phi_good, bins=100, normed=True)
    hist_all, xedg_all, yedg_all = np.histogram2d(wind_data.theta, wind_data.phi, bins=100, normed=True)

    # Build the plot
    fig = plt.figure(figsize=(22, 8))

    limits_x = (89, 98) if DEGREES else (1.56, 1.72)
    limits_y = (83, 92) if DEGREES else (1.45, 1.61)

    # Only bad points
    fig.add_subplot(141, axisbg='black')
    plt.title("Anomalous points")
    plt.xlabel(r"$\theta$", fontsize=30)
    plt.ylabel(r"$\varphi$", fontsize=30, rotation=0)
    plt.xlim(*limits_x)
    plt.ylim(*limits_y)
    im1 = plt.imshow(hist_bad, interpolation='nearest', origin='low', cmap='gnuplot', alpha=1.,
                     extent=[xedg_bad[0], xedg_bad[-1], yedg_bad[0], yedg_bad[-1]])
    # Only good points
    fig.add_subplot(142, axisbg='black')
    plt.title("Non-anomalous points")
    plt.xlabel(r"$\theta$", fontsize=30)
    plt.ylabel(r"$\varphi$", fontsize=30, rotation=0)
    plt.xlim(*limits_x)
    plt.ylim(*limits_y)
    im2 = plt.imshow(hist_good, interpolation='nearest', origin='low', cmap='cubehelix', alpha=1.,
                     extent=[xedg_good[0], xedg_good[-1], yedg_good[0], yedg_good[-1]])

    # Both good and bad points
    fig.add_subplot(143, axisbg='black')
    plt.title("All points")
    plt.xlabel(r"$\theta$", fontsize=30)
    plt.ylabel(r"$\varphi$", fontsize=30, rotation=0)
    plt.xlim(*limits_x)
    plt.ylim(*limits_y)
    im1 = plt.imshow(hist_bad, interpolation='nearest', origin='low', cmap='gnuplot', alpha=1.,
                     extent=[xedg_bad[0], xedg_bad[-1], yedg_bad[0], yedg_bad[-1]])

    im2 = plt.imshow(hist_good, interpolation='nearest', origin='low', cmap='cubehelix', alpha=0.5,
                     extent=[xedg_good[0], xedg_good[-1], yedg_good[0], yedg_good[-1]])

    fig.add_subplot(144, axisbg='black')
    plt.title("All points")
    plt.xlabel(r"$\theta$", fontsize=30)
    plt.ylabel(r"$\varphi$", fontsize=30, rotation=0)
    plt.xlim(*limits_x)
    plt.ylim(*limits_y)
    im1 = plt.imshow(hist_all, interpolation='nearest', origin='low', cmap='gnuplot', alpha=0.9,
                     extent=[xedg_all[0], xedg_all[-1], yedg_all[0], yedg_all[-1]])

    plt.show()


def check_one(holder):
    # Limit the data to the interesting interval
    #holder.cut_time(2700., 3000.)

    temp_data = dataholders.TempData(holder.time, holder.T1)
    temp_data.preprocess(filters.median_filter, 21)
    temp_data.process(heuristics.base_median_heuristic, 5)
    temp_data.detect(.015)

    temp_data.plot(True)


if __name__ == "__main__":
    # Load the data
    temp_path = 'data/raw/uft_flight07.mat'
    wind_path = 'data/raw/actos_flight07.mat'

    data_holder = dataholders.TempWindData(temp_path, wind_path)
    #3950 4150
    plot_histograms(data_holder)
