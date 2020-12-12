# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:52:43 2020

@author: Kiera
"""
# Main Imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Recursive Feature Cross
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
# Principal Feature Analysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from sklearn import manifold

def recursiveFeatureCross(X, y):
    
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy')
    rfecv.fit(X, y)
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    return rfecv.ranking_



class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]

        sc = StandardScaler()
        X = sc.fit_transform(X)

        pca = PCA(n_components=self.q).fit(X)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]
        
        
        

def swiss_roll(X, color):
    
    print("Computing LLE embedding")
    X_r, err = manifold.locally_linear_embedding(X, n_neighbors=200,
                                                 n_components=3)
    print("Done. Reconstruction error: %g" % err)
    
    #----------------------------------------------------------------------
    # Plot result
    
    # Create Plot and Axes
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1])
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title("Original data")
    
    # New Plot and Axes
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1])
    ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title('Projected data')