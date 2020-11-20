# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:36:52 2020

@author: Kiera
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans , OPTICS, cluster_optics_dbscan
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.pairwise import pairwise_distances_argmin 

def kmeans2(X, n=10):
    """(2D) Using the K-Means clustering algorithm, group the data X into n clusters

    Args:
        X (2xN float numpy array): Data as columns
        n (int, optional): Number of Clusters. Defaults to 10.

    Returns:
        (1xN integer numpy array): Generated Classification Labels
    """

    
    # Define model
    kme = KMeans(init ='k-means++', n_clusters=n)
    
    
    
    # Fit data with model
    kme.fit(X)
    labels = kme.labels_


    # Create Plot and Axes
    plt.figure(figsize=(10, 10))
    
    # Plot data
    plt.scatter(X[:, 0], X[:, 1], c=labels.astype(np.float), edgecolor='k')

    # Axes stuff again
    plt.title("KMeans Classification")
    plt.axis("tight")
    
    # Make y prediction
    y_pred = kme.predict(X)
    
    # Return prediction
    return y_pred
    
def miniBatchKMeans2(X, n=10, b=100):
    """(2D) Using the MiniBatch K-Means Clustering algorithm, group the data X into n clusters with a batch size of b

    Args:
        X (2xN float numpy array): Data as column
        n (int, optional): Number of Clusters. Defaults to 10.
        b (int, optional): Batch Sizing. Defaults to 100.

    Returns:
        1xN integer numpy array: Generated Classification Labels
    """

    # perform the mini batch K-means 
    mbk = MiniBatchKMeans(init ='k-means++', n_clusters = n, 
                          batch_size = b, n_init = 10, 
                          max_no_improvement = 10, verbose = 0) 
      
    # Fit model with data
    mbk.fit(X) 
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis = 0) 
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)     
    
    # Create Plot and Axes
    plt.figure(figsize=(10, 10))
    
    # Plot data
    plt.scatter(X[:, 0], X[:, 1], c=mbk_means_labels.astype(np.float), edgecolor='k')

    # Axes stuff again
    plt.title("MiniBatch KMEANS Classification")
    plt.axis("tight")
    
    # Make y prediction
    y_pred = mbk.predict(X)
    
    # Return prediction
    return y_pred
    
def birch2(X, n=10):
    """ (2D)Using the Birch clustering algorithm, group the data X into n clusters

    Args:
        X (2xN float numpy array): Data as columns
        n (int, optional): Number of Clusters. Defaults to 10.

    Returns:
        (1xN integer numpy array): Generated Classification Labels
    """
    
    # Define Model
    bir = Birch(n_clusters=n)

    # Fit data with model
    bir.fit(X)
    labels = bir.labels_

    # Create Plot and Axes
    plt.figure(figsize=(10, 10))
    
    # Plot data
    plt.scatter(X[:, 0], X[:, 1], c=labels.astype(np.float), edgecolor='k')

    # Axes stuff again
    plt.title("Birch Classification")
    plt.axis("tight")
    
    # Make y prediction
    y_pred = bir.predict(X)
    
    # Return prediction
    return y_pred
    