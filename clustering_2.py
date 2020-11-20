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

def kmeans2(X, n):
    # X : np matrix : data values
    # n : n clusters
    
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
    
def miniBatchKMeans2(X, n, b):
    # X : np matrix : data values
    # n : n clusters    
    # b : batch size
    
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
    
def birch2(X, n):
    # X : np matrix : data values
    # n : n clusters
    
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
    
    
def optics2(X, n_points_per_cluster = 250):
    # X : np matrix : data values
    # n : n points per clusters : default = 250
    
    
    clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)

    # Run the fit
    clust.fit(X)
    
    labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=0.5)
    labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=2)
    
    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])
    
    # Reachability plot
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')
    
    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = X[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')
    
    # DBSCAN at 0.5
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    for klass, color in zip(range(0, 6), colors):
        Xk = X[labels_050 == klass]
        ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
    ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')
    
    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = X[labels_200 == klass]
        ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
    ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')
    
    plt.tight_layout()
    plt.show()
    
    
def dendogram2(X):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    
    # fit model
    model = model.fit(X)
    
    plt.figure(figsize=(10,10))
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    
    
# FROM SKLEARN DOCS https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    
    
    
