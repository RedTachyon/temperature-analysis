import json

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

import dataholders as dh
import utils


def prepare_points(path='../data/points.txt'):
    """
    Parses the output of label app to a dataframe with timestamps of all jump points, including
    the nearest points from the original time vector, converted to milliseconds
    """
    
    with open('data/points.txt') as f:
        lines = f.readlines()
    
    lines = list(map(lambda x: x.strip(), lines)) # remove extra newlines
    lines = list(map(json.loads, lines))
    lines = list(filter(lambda x: x['mode'] == 'jump', lines)) # choose only jump points
    
    df = pd.DataFrame(lines)
    
    # Cast everything to numeric datatypes
    df['noiseStep'] = df['noiseStep'].astype(int)
    df['thermometer'] = df['thermometer'].astype(int)
    df['value'] = df['value'].astype(float)
    df['value'] = (1000 * df['value']).astype(int) # Storing timestamps in int's for easier comparing
    
    df.value = df.value.apply(lambda x: x + 1 if x % 10 != 0 else x) # Fix a dumb glitch
    
    # The problem with these datasets is that the timestamps in the original time vector
    # always end in .005 (in seconds) or 5 (in milliseconds, like they are represented here)
    # while points end with .000 or 0 (respectively).
    # To alleviate that, I keep track of the nearest points to each jump.
    df['value_after']  = df.value + 5
    df['value_before'] = df.value - 5
    
    return df

def prepare_data(points_path='../data/points.txt', data_path='data/data_flight16.mat'):
    """
    Reads the marked jump points and the time and temperature vectors, with time measured in milliseconds.
    """
    
    df = prepare_points(points_path)
    all_data = loadmat(data_path)
    time, lowT, upT = all_data['time_av'].ravel(), all_data['lowT_av'].ravel(), all_data['upT_av'].ravel()
    
    time = (time*1000).astype(int) # Convert to milliseconds
    time = np.round(time, -1) + 5 # Fix some points that end in 4 instead of 5
    
    low_labels = np.logical_or(np.in1d(time, df[df.thermometer == 0].value_after), 
                               np.in1d(time, df[df.thermometer == 0].value_before)
                              ).astype(int)
    
    up_labels = np.logical_or(np.in1d(time, df[df.thermometer == 1].value_after), 
                              np.in1d(time, df[df.thermometer == 1].value_before)
                             ).astype(int)
    return time, low_labels, up_labels, lowT, upT

def smear_labels(a: np.ndarray, size: int) -> np.ndarray:
    """
    Gives positive labels to points within $size points of actual positive labels.
    
    Args:
        a: np.array, labels array
        size: int, how far away the labels are smeared
    """
    
    pad = size # Amount of zero's to add on both sides
    a_padded = np.concatenate((np.zeros(pad), a, np.zeros(pad)))
    rolled = utils.rolling_window(a_padded, size*2 + 1)
    
    return rolled.max(1).astype(a.dtype)

def generate_positive(labels, time, temperature):
    """
    Generates time and temperature vectors of length 20 (to be variablefied), containing a jump.
    """
    time_data        = []
    temperature_data = []
    
    for i, val in tqdm(enumerate(labels), total=len(labels)):
        window = labels[i:i+20] # 20: how many points around a jump
        hit = np.array([1,1])
        l_l, l_r = i+4, i+6   # jump is on the left   of the interval
        c_l, c_r = i+9, i+11  # jump is in the center of the interval
        r_l, r_r = i+14, i+16 # jump is on the right  of the interval

        if np.array_equal(labels[l_l:l_r], hit) or np.array_equal(labels[c_l:c_r], hit) or np.array_equal(labels[r_l:r_r], hit):
            time_data.append(time[i:i+20])
            temperature_data.append(temperature[i:i+20])

    return np.array(time_data), np.array(temperature_data)

def generate_negative(labels, time, temperature):
    """
    Generates time and temperature vectors of length 20, hopefully not containing a jump, in a safe distance from actual jumps.
    """
    labels = labels.copy()

    time_data        = []
    temperature_data = []

    for i, val in tqdm(enumerate(labels), total=len(labels)):
        window = labels[i:i+50]
        if window.max() == 0 and i + 50 < len(labels):
            time_data.append(time[i+15:i+35])
            temperature_data.append(temperature[i+15:i+35])
            labels[i:i+50] = 1
    
    return np.array(time_data), np.array(temperature_data)