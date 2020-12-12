# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:48:08 2020

@author: Kiera
"""

## THIS IS NOT DETERMINISTIC AND AS SUCH WILL NOT ALWAYS YEILD THE SAME GRAPH

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
df = pd.read_csv("../data/SCC403CWWeatherData.txt")

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
mask = df['precipipitation'] > 0
df_raining = df[mask]
df1 = df[~mask]

# Drop non-required columns
df1 = df1.drop(['precipipitation', 'snowfall'], axis=1)


# SCALING
# Scale all columns - return float64 array
npscaled = preprocessing.scale(df1)

# Show mean and deviance
npscaled.mean(axis=0)
npscaled.std(axis=0)

dfcolumns = ['temp.min', 'temp.max', 'temp.mean',
              'hum.min', 'hum.max', 'hum.mean',
              'press.min', 'press.max', 'press.mean',
              'sunshine',
              'gust.min', 'gust.max', 'gust.mean',
              'speed.min', 'speed.max', 'speed.mean']
df2 = pd.DataFrame(npscaled, columns=dfcolumns)


kbin = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
sunshine = kbin.fit_transform(df2[["sunshine"]])
df2["sunshine"] = sunshine

df_numpy = df2.to_numpy()

metric_values=[]

for rank_r in range(2, 15):

    # Parameters:
    #rank_r = 5	    	# Number of L1-norm principal components.
    num_init = 10 		# Number of initializations.
    print_flag = False	# Print decomposition statistics (True/False).
    	
    	
    # Call the L1-norm PCA function.
    Q, B, vmax= l1pca(df_numpy, rank_r, num_init, print_flag)
    
    metric_values.append([rank_r, vmax])
    
    #print(Q) # Print the calculated subspace matrix.
    
plt.figure(figsize=(5, 4))
plt.plot(*zip(*metric_values))
plt.xlabel('Number of L1-norm principal components')
plt.ylabel('Metric Value')
plt.savefig("figures/L1-Norm", facecolor="w", edgecolor="w",
                dpi=300)
    