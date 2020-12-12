# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:46:48 2020

@author: Kiera
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import svm, metrics, model_selection
from sklearn.neighbors import NearestCentroid


def basic_classification(X, y):
    """Taking a numpy array and classification labels, splitting into training and testing data using a 80/20 
    split, then generating a model and using the testing data to predict new classification labels and 
    determining the model accuracy 

    Args:
        X (MxN float numpy array):  Data as columns
        y (1xN integer numpy array): storing classification labels
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80,test_size=0.20)
    
    #Create a svm Classifier
    clf = svm.SVC(kernel='rbf') # Linear Kernel
    
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    # Determine Accuracy?
    print("Accuracy: {0}".format(metrics.accuracy_score(y_test, y_pred)))
    
    # Print Class Report 
    print(metrics.classification_report(y_test, y_pred))
    
    
    
def centroid(X, y, n = 15, h = 0.02):
    """Taking a dataset and classification labels, performing Nearest Centroid Classification and then 
    plotting the data with decision boundaries for each class

    Args:
        X (MxN float numpy array): Data as columns
        y (1xN integer numpy array): Classification Labels
        n (int, optional): Number of Neighbours. Defaults to 15.
        h (float, optional): Mesh Stepsize. Defaults to 0.02.
    """ 

    # Do for no shrinkage and also .2 shrinkage
    for shrinkage in [.2]:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = NearestCentroid(shrink_threshold=shrinkage)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        print("Shrinkage : {0} Average : {1}".format(shrinkage, np.mean(y == y_pred)))
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired,
                    edgecolor='k', s=20)
        plt.title("Multi-Class NN classification (shrink_threshold={0})".format(shrinkage))
        plt.axis('tight')
    
    plt.show()
    
    
def svm_rbf(X, y, h = .02):
    """Taking a dataset and classification labels, using a Support Vector Machine to classify a sample. 
    Plotting the data with decision surfaces and the support vectors

    Args:
        X (MxN float numpy array): Data as columns
        y (1xN integer numpy array): Classification labels
        h (float, optional): Mesh Stepsize. Defaults to .02.
    """
    
    # we create an instance of SVM and fit out data.
    clf = svm.SVC(kernel="rbf")
    clf.fit(X, y)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    plt.title('Multi-Class classification using Support Vector Machine using RBF')
    plt.axis('tight')
    plt.show()