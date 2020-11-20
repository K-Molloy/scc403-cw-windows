# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:52:46 2020

@author: Kiera
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def pca_transform(df, n):
    """Takes a pandas dataframe or numpy array and performs PCA reduction into n dimensions. Returning a tuple of XxN numpy array and XxN pandas dataframe

    Args:
        df (datafram): Data stored in columns
        n (integer): Number of PCA Components

    Returns:
        tuple(numpy array, dataframe): PCA Reduced Data
    """
    # df : pd dataframe : data
    # n : int : pca components
    
    # Define PCA transform
    pca = PCA(n_components=n)
    # Prepare Tranformation by using dataset
    pca.fit(df)
    # Find the component coefficients
    coeff = pca.components_
    # Apply the transformation to the dataset
    T = pca.transform(df)
    # change 'T' to Pandas-DataFrame
    pandaT = pd.DataFrame(T)
    
    # return tuple numpy array, pandas dataframe
    return T, pandaT

def pca_analysis(df):
    """Takes a pandas dataframe or numpy array and calculates the variance for each PCA(n) where n is 0 -> size of data. Plotting a graph with a horizontal line at 90% threshold

    Args:  
        df (dataframe): Data stored in columns
    """
    
    # df : pd dataframe : data
    
    # Define model
    pca = PCA().fit(df)
    
    # Initialise new figure
    plt.figure()    
    # Plot variances
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # Plot Horizontal Line
    plt.axhline(y=0.9, color='r', linestyle='-')
    # Axis Labels
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    
    
def pca_3_plot_basic(df):
    """Takes a pandas dataframe or numpy array and calculates a 3 dimenionsional PCA and creates a scatterplot

    Args:
        df (dataframe): Data stored in columns
    """
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    
    X_reduced = PCA(n_components=3).fit_transform(df)
    # Initialise figure and axes
    fig = plt.figure(1, figsize=(10, 10))
    ax = Axes3D(fig, elev=-150, azim=110)
    
    # Plot scatter
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], edgecolor='k', s=40)
    ax.set_title("First three PCA directions")
    # Title, Labels and axes labels
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    
def pca_3_labels(df, y):
    """Takes a pandas dataframe or numpy array and calculated a 3 dimensional PCA, and using classification labers created a scatterplot with colours according to class

    Args:
        df (dataframe): Data stored in columns
        y (array): 1xN array of classification integers
    """
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    
    X_reduced = PCA(n_components=3).fit_transform(df)
    # Initialise figure and axes
    fig = plt.figure(1, figsize=(10, 10))
    ax = Axes3D(fig, elev=-150, azim=110)
    
    # Plot scatter
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')
    ax.set_title("First three PCA directions")
    # Title, Labels and axes labels
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])