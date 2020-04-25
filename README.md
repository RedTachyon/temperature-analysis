# Temperature data analysis

This repository holds the code for analyzing the data from the Azores 2017 campaign which was the main point of my Bachelor's thesis. It involves statistical analysis of the data, as well as ML modeling and prediction

uft\_flightXX.mat and actos\_flightXX.mat files must be placed in the data/raw/ directory.

## Basic usage: 

Install the required packages with pip3 install -r requirements.txt

run main.py (either with ./main.py or python3 main.py) to see the wind histograms. 

heuristics.py contains detection algorithms

filters.py contains filters for preprocessing the temperature data

dataholders.py contains classes structures for manipulating temperature and wind data

utils.py contains various helper functions 
