#!/bin/python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utils
import heuristics
import filters
import dataholders

plt.rcParams['figure.figsize'] = (18, 6)

DEGREES = True

if __name__ == "__main__":
    # Load the data
    temp_path = 'data/raw/uft_flight07.mat'
    wind_path = 'data/raw/actos_flight07.mat'
    holder = dataholders.TempWindData(temp_path, wind_path)

    # Limit the data to the interesting interval
    holder.cut_time(2700., 3000.)

    # Extract and process the temperature data
    temp_data = dataholders.TempData(holder.time, holder.T1)
    temp_data.preprocess(filters.median_filter, 21)
    temp_data.process(heuristics.base_median_heuristic, 7)
    temp_data.detect(.015)

    # Extract and process the wind data
    wind_data = dataholders.WindData(holder.v1, holder.v2, holder.v3, degrees=DEGREES)

    print("Percentage of anomalies: %.2f%%" % (temp_data.mask.mean()*100))

    # Select the angles with and without anomalies

    theta_bad = wind_data.theta[temp_data.mask]
    phi_bad = wind_data.phi[temp_data.mask]

    theta_good = wind_data.theta[~temp_data.mask]
    phi_good = wind_data.phi[~temp_data.mask]

    # Compute the histograms
    hist_bad, xedg_bad, yedg_bad = np.histogram2d(theta_bad, phi_bad, bins=100)
    hist_good, xedg_good, yedg_good = np.histogram2d(theta_good, phi_good, bins=100)

    # Build the plot
    fig = plt.figure(figsize=(22, 8))

    limits_x = (89, 98) if DEGREES else (1.56, 1.72)
    limits_y = (83, 92) if DEGREES else (1.45, 1.61)
    # Only bad points
    fig.add_subplot(131, axisbg='black')
    plt.title("Anomalous points")
    plt.xlim(*limits_x)
    plt.ylim(*limits_y)
    im1 = plt.imshow(hist_bad, interpolation='nearest', origin='low', cmap='gnuplot', alpha=1.,
                     extent=[xedg_bad[0], xedg_bad[-1], yedg_bad[0], yedg_bad[-1]])
    # Only good points
    fig.add_subplot(132, axisbg='black')
    plt.title("Non-anomalous points")
    plt.xlim(*limits_x)
    plt.ylim(*limits_y)
    im2 = plt.imshow(hist_good, interpolation='nearest', origin='low', cmap='cubehelix', alpha=1.,
                     extent=[xedg_good[0], xedg_good[-1], yedg_good[0], yedg_good[-1]])

    # Both good and bad points
    fig.add_subplot(133, axisbg='black')
    plt.title("All points")
    plt.xlim(*limits_x)
    plt.ylim(*limits_y)
    im1 = plt.imshow(hist_bad, interpolation='nearest', origin='low', cmap='gnuplot', alpha=1.,
                     extent=[xedg_bad[0], xedg_bad[-1], yedg_bad[0], yedg_bad[-1]])

    im2 = plt.imshow(hist_good, interpolation='nearest', origin='low', cmap='cubehelix', alpha=.5,
                     extent=[xedg_good[0], xedg_good[-1], yedg_good[0], yedg_good[-1]])

    plt.show()
