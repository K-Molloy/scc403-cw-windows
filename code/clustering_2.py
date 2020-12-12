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
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics.pairwise import pairwise_distances_argmin 

from sklearn import metrics
from sklearn.metrics import pairwise_distances, davies_bouldin_score

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

    cluster_plot(X, labels, "K-Means", n)
    
    # Make y prediction
    y_pred = kme.predict(X)
    
    # Return prediction
    return y_pred
    
def mini_batch_kmeans2(X, n=10, b=100):
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
    labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)     
    

    cluster_plot(X, labels, "Mini-K", n)
    
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


    cluster_plot(X, labels, "Birch", n)
    
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
    labels = spc.labels_

    cluster_plot(X, labels, "spectral", n)
    
    # Make y prediction
    y_pred = spc.fit_predict(X)
    
    # Return prediction
    return y_pred

def vbgmm(X, n=10):
    
    gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(X)
    return gmm.predict(X)

def met(X, y):
    
    # Silhouette Score
    sil = metrics.silhouette_score(X, y, metric='euclidean')
    # Calinkski-Harabasz Index
    ch = metrics.calinski_harabasz_score(X, y)
    # Davies-Bouldin Index
    db = metrics.davies_bouldin_score(X, y)
    
    return sil, ch, db


def cluster_plot(X, labels, method, n):
    
    fig, axs = plt.subplots(3, 2, figsize=(5, 4), constrained_layout=True)
    fig.suptitle("{0} Classification: {1} Clusters".format(method, n))
    # Plot data
    axs[0,0].scatter(X[:, 0], X[:, 1], c=labels.astype(np.float), edgecolor='k')
    axs[0,1].scatter(X[:, 0], X[:, 2], c=labels.astype(np.float), edgecolor='k')
    axs[1,0].scatter(X[:, 0], X[:, 3], c=labels.astype(np.float), edgecolor='k')
    axs[1,1].scatter(X[:, 0], X[:, 4], c=labels.astype(np.float), edgecolor='k')
    axs[2,0].scatter(X[:, 0], X[:, 5], c=labels.astype(np.float), edgecolor='k')
    axs[2,1].scatter(X[:, 1], X[:, 5], c=labels.astype(np.float), edgecolor='k')
    axs[0,0].set_title('PC 1 x PC 2')
    axs[0,1].set_title('PC 1 x PC 3')
    axs[1,0].set_title('PC 1 x PC 4')
    axs[1,1].set_title('PC 1 x PC 5')
    axs[2,0].set_title('PC 1 x PC 6')
    axs[2,1].set_title('PC 2 x PC 6')
    fig.savefig("figures/{0}-{1}".format(method, n), facecolor="w", edgecolor="w",
                dpi=300)
    

    