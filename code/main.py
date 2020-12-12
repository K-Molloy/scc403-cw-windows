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

# Quick Metric thing
from sklearn.metrics.cluster import contingency_matrix

# Custom Functions
from feature_selection import recursiveFeatureCross, PFA, swiss_roll
from l1pca import l1pca
from processing import pca_analysis, pca_3_plot_basic, pca_transform, pca_3_labels, pca_variance
from clustering_3 import kmeans, mini_batch_kmeans, birch, optics, create_dendrogram
from clustering_2 import kmeans2, mini_batch_kmeans2, birch2, spectral, met
from classification import basic_classification, centroid, svm_rbf
from idrk import compare_clustering


## READING IN DATA    

# Read Data set
df = pd.read_csv("../data/SCC403CWWeatherData.txt", header=None)

## EXPLORATORY ANALYSIS

df.columns = ['temp.min', 'temp.max', 'temp.mean',
              'hum.min', 'hum.max', 'hum.mean',
              'press.min', 'press.max', 'press.mean',
              'precipipitation', 'snowfall', 'sunshine',
              'gust.min', 'gust.max', 'gust.mean',
              'speed.min', 'speed.max', 'speed.mean']

## PRE-PROCESSING

# Check for missing values
df.isna().sum() # there are none 

# SPLIT DATASET

# Remove those that have snow 
mask = df['snowfall'] > 0
df_snowing = df[mask]
df1 = df[~mask]

# Remove those that have rain
mask = df1['precipipitation'] > 0
df_raining = df1[mask]
df_standard = df1[~mask]

# Drop non-required columns
df_standard = df_standard.drop(['precipipitation', 'snowfall'], axis=1)
df_raining = df_raining.drop(['snowfall'], axis=1)

# Output CSVs before standardisation
df_snowing.to_csv('csv/df_snowing.csv', encoding='utf-8')
df_raining.to_csv('csv/df_raining.csv', encoding='utf-8')
df_standard.to_csv('csv/df_standard.csv', encoding='utf-8')



############################################################
## STANDARD SUBSET
############################################################

# SCALING
# Scale all columns - return float64 array
# npscaled = preprocessing.scale(df_standard)

# # Show mean and deviance
# npscaled.mean(axis=0)
# npscaled.std(axis=0)

# # Rename columns
# dfcolumns = ['temp.min', 'temp.max', 'temp.mean',
#               'hum.min', 'hum.max', 'hum.mean',
#               'press.min', 'press.max', 'press.mean',
#               'sunshine',
#               'gust.min', 'gust.max', 'gust.mean',
#               'speed.min', 'speed.max', 'speed.mean']
# df_standard = pd.DataFrame(npscaled, columns=dfcolumns)

# KBin Discretise
kbin = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
sunshine = kbin.fit_transform(df_standard[["sunshine"]])
df_standard["sunshine"] = sunshine
# Convert to numpy for everything else
df_numpy = df_standard.to_numpy()
df_numpy = preprocessing.scale(df_numpy)

## INITIAL PFA Analysis

pca_analysis(df_numpy)

# PFA
pfa = PFA(n_features=6)
pfa.fit(df_numpy)
X = pfa.features_
column_indices = pfa.indices_

# L1-Norm PCA Transform
k = 6
num_init = 10
print_flag = False
Q, B, vmax = l1pca(df_numpy, k, num_init, print_flag)

Q = preprocessing.scale(Q)

## CLUSTERING

compare_clustering(Q)

for i in [3, 4, 5]:
    # kmeans clustering
    y1 = kmeans2(Q, i)
    # birch clustering
    y2 = birch2(Q, i)
    # mini-batch k-means clustering
    y3 = mini_batch_kmeans2(Q, i, 160)
    # spectral
    y4 = spectral(Q, i)
    
    # Calculate cluster differences
    print("CLustering {0}".format(i))
    basic_classification(Q, y1)
    basic_classification(Q, y2)
    basic_classification(Q, y3)
    basic_classification(Q, y4)
    
    if i == 3:
        yfin = y1


# Contigency Matrices
cmp1 =  contingency_matrix(y1, y2)
cmp2 =  contingency_matrix(y1, y3)
cmp3 =  contingency_matrix(y1, y4)
cmp4 =  contingency_matrix(y2, y3)


# Plot Dendogram
create_dendrogram(Q)
swiss_roll(Q, y3)
ranking = recursiveFeatureCross(Q, y1)

# Calculate cluster differences
basic_classification(Q, y1)
basic_classification(Q, y2)
basic_classification(Q, y3)
basic_classification(Q, y4)

df_standard['label'] = yfin

############################################################
## RAINING SUBSET
############################################################

# SCALING
# Scale all columns - return float64 array
# npscaled = preprocessing.scale(df_raining)

# # Show mean and deviance
# npscaled.mean(axis=0)
# npscaled.std(axis=0)

# # Rename columns
# dfcolumns = ['temp.min', 'temp.max', 'temp.mean',
#               'hum.min', 'hum.max', 'hum.mean',
#               'press.min', 'press.max', 'press.mean',
#               'precipipitation','sunshine',
#               'gust.min', 'gust.max', 'gust.mean',
#               'speed.min', 'speed.max', 'speed.mean']
# df_raining = pd.DataFrame(npscaled, columns=dfcolumns)

# KBin Discretise
kbin = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
sunshine = kbin.fit_transform(df_raining[["sunshine"]])
df_raining["sunshine"] = sunshine
rain = kbin.fit_transform(df_raining[["precipipitation"]])
df_raining["precipipitation"] = rain
# Convert to numpy for everything else
df_numpy = df_raining.to_numpy()
df_numpy = preprocessing.scale(df_numpy)

## INITIAL PFA Analysis

# PFA
pfa = PFA(n_features=6)
pfa.fit(df_numpy)
X = pfa.features_
column_indices = pfa.indices_

# L1-Norm PCA Transform
k = 6
num_init = 10
print_flag = False
Q, B, vmax = l1pca(df_numpy, k, num_init, print_flag)

Q = preprocessing.scale(Q)

## CLUSTERING

compare_clustering(Q)

for i in [5]:
    # kmeans clustering
    y1 = kmeans2(Q, i)
    # birch clustering
    y2 = birch2(Q, i)
    # mini-batch k-means clustering
    y3 = mini_batch_kmeans2(Q, i, 200)
    # spectral
    y4 = spectral(Q, i)
    
    # Calculate cluster differences
    print("CLustering {0}".format(i))
    basic_classification(Q, y1)
    basic_classification(Q, y2)
    basic_classification(Q, y3)
    basic_classification(Q, y4)


# Contigency Matrices
cmp1 =  contingency_matrix(y1, y2)
cmp2 =  contingency_matrix(y1, y3)
cmp3 =  contingency_matrix(y1, y4)
cmp4 =  contingency_matrix(y2, y3)


# Plot Dendogram
create_dendrogram(Q)
swiss_roll(Q, y3)
ranking = recursiveFeatureCross(Q, y1)

# Calculate cluster differences
basic_classification(Q, y1)
basic_classification(Q, y2)
basic_classification(Q, y3)
basic_classification(Q, y4)


# Data Labelling with the best accuracy
df_raining['label'] = y3 + 3

df_snowing['label'] = 8

# Merging Frames
frames = [df_standard, df_raining, df_snowing]

result = pd.concat(frames)
result = result.sort_index()

# Re-Extract Labels and put onto original dataset
y = result['label'].to_numpy()
df['label'] = y

# Save to CSV for re-usability
df.to_csv('csv/df_final.csv', encoding='utf-8')

    
    