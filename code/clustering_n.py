# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:36:52 2020

@author: Kiera
"""

import numpy as np
import pandas as pd

from sklearn.cluster import Birch, KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances_argmin 

def kmeans(X, n):
    """ Using the K-Means clustering algorithm, group the data X into n clusters

    Args:
        X (MxN float numpy array): Data as columns
        n (int, optional): Number of Clusters. Defaults to 10.

    Returns:
        (1xN integer numpy array): Generated Classification Labels
    """
    
    # Define model
    kme = KMeans(init ='k-means++', n_clusters=n)
        
    # Fit data with model
    kme.fit(X)

    # Make y prediction
    y_pred = kme.predict(X)
    
    # Return prediction
    return y_pred
    
def mini_batch_kmeans(X, n, b):
    """ Using the MiniBatch K-Means Clustering algorithm, group the data X into n clusters with a batch size of b

    Args:
        X (MxN float numpy array): Data as column
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
    
    
    # Make y prediction
    y_pred = mbk.predict(X)
    
    # Return prediction
    return y_pred
    
def birch(X, n):
    """ Using the Birch clustering algorithm, group the data X into n clusters

    Args:
        X (MxN float numpy array): Data as columns
        n (int, optional): Number of Clusters. Defaults to 10.

    Returns:
        (1xN integer numpy array): Generated Classification Labels
    """
    
    bir = Birch(n_clusters=n)

    # Fit data with model
    bir.fit(X)
    # Make y prediction
    y_pred = bir.predict(X)
    
    # Return prediction
    return y_pred

def spectral(X, n=10):
    
    # perform the mini batch K-means 
    spc = SpectralClustering(n_clusters = n,
                             assign_labels="discretize",
                             random_state=0)
      
    # Fit data with model
    spc.fit(X)
    
    # Make y prediction
    y_pred = spc.fit_predict(X)
    
    # Return prediction
    return y_pred

    
    
    
    
