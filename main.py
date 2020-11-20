# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:48:08 2020

@author: Kiera
"""

# Big Package Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc

# Niche Package Imports
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA

# Custom Functions
from processing import pca_analysis, pca_3_plot_basic, pca_transform, pca_3_labels
from clustering_3 import kmeans, miniBatchKMeans, birch, optics, dendogram
from clustering_2 import kmeans2, miniBatchKMeans2, birch2, optics2, dendogram2
from classification import basic_classification, centroid, svm_rbf

## READING IN DATA    

# Read Data set
df = pd.read_csv("SCC403CWWeatherData.txt")

# Assign Column Names
df.columns = ['temp.min', 'temp.max', 'temp.mean',
              'hum.min', 'hum.max', 'hum.mean',
              'press.min', 'press.max', 'press.mean',
              'precipipitation', 'snowfall', 'sunshine',
              'gust.min', 'gust.max', 'gust.mean',
              'speed.min', 'speed.max', 'speed.mean']


## EXPLORATORY ANALYSIS

# Some initial plots
df[['temp.min', 'temp.max', 'temp.mean']].plot()
df[['hum.min', 'hum.max', 'hum.mean']].plot()
df[['press.min', 'press.max', 'press.mean']].plot()
df[['gust.min', 'gust.max', 'gust.mean']].plot()
df[['speed.min', 'speed.max', 'speed.mean']].plot()

## PRE-PROCESSING

# Check for missing values
df.isna().sum() # there are none 

dfscaled = preprocessing.scale(df)

# Show mean and deviance
dfscaled.mean(axis=0)
dfscaled.std(axis=0)

# Determine good amount of components
pca_analysis(dfscaled)
pca_3_plot_basic(dfscaled)

# PCA Transform
T, pandaT = pca_transform(dfscaled, 3)

## CLUSTERING

# kmeans clustering
y1 = kmeans(T, 10)
# birch clustering
y2 = birch(T, 10)
# mini-batch k-means clustering
y3 = miniBatchKMeans(T, 10, 100)
# create an OPTICS graph
optics(T, 100)

# Plot Dendogram
dendogram(T)

# Calculate cluster differences
metrics.adjusted_rand_score(y1, y2)
metrics.adjusted_rand_score(y1, y3)
metrics.adjusted_rand_score(y2, y3)


## CLASSIFICATION


basic_classification(T, y1)
basic_classification(T, y2)
basic_classification(T, y3)


## 2D Version

# PCA Transform
T2, pandaT2 = pca_transform(dfscaled, 2)

# kmeans clustering
y1_2 = kmeans2(T2, 10)
# birch clustering
y2_2 = birch2(T2, 10)
# mini-batch k-means clustering
y3_2 = miniBatchKMeans2(T2, 10, 100)
# create an OPTICS graph
optics2(T2, 100)

# Plot Dendogram
dendogram2(T2)

# Calculate cluster differences
metrics.adjusted_rand_score(y1_2, y2_2)
metrics.adjusted_rand_score(y1_2, y3_2)
metrics.adjusted_rand_score(y2_2, y3_2)

centroid(T2, y1_2)
svm_rbf(T2, y1_2)



    
    