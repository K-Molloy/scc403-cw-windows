# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 23:38:16 2020

@author: Kiera
"""
import numpy as np
import matplotlib.pyplot as plt

from clustering_2 import met
from clustering_n import kmeans, mini_batch_kmeans, birch, spectral

def compare_clustering(X, max_clusters=20, batch_size=200):
    
    # Iterate over 1 to 20 clusters
    # Calculate each method
    
    kmeans_score=[]
    birch_score=[]
    minik_score=[]
    spectral_score=[]
    
    print("Calculating Clusters:")
    
    for i in range(2, max_clusters):
        
        # Clustering Methods
        try:
            clustering = {
                "kmeans" : kmeans(X, i),
                "birch" : birch(X, i),
                "minik" : mini_batch_kmeans(X, i, batch_size),
                "spectral" : spectral(X, i)
                }
        except:
            print("Error Calculating Methods")
        
        kmeans_score.append(met(X, clustering["kmeans"]))
        birch_score.append(met(X, clustering["birch"]))
        minik_score.append(met(X, clustering["minik"]))
        spectral_score.append(met(X, clustering["spectral"]))  
        
        print(".", end="")
        
    print("Done")
    compare_plots(kmeans_score, birch_score, minik_score, spectral_score)
            
    return kmeans_score, birch_score, minik_score



def compare_plots(kmeans, birch, minik, spectral):
    
    kmeans = extract_score(kmeans)
    birch = extract_score(birch)
    minik = extract_score(minik)
    spectral = extract_score(spectral)
    
    
    
    make_plot(kmeans, "K-Means")
    make_plot(birch, "Birch")
    make_plot(minik, "K-Means Mini Batch")
    make_plot(spectral, "Spectral")
        
def make_plot(scores, method):
    
    x=np.arange(2, 20)
    
    plt.figure(figsize=(5, 4))
    fig, ax = plt.subplots()
    ax.plot(x,scores[0],c='r',marker="^",ls='--',label='Silhouette')
    ax.plot(x,scores[2],c='r',marker="o",ls='-',label='Davies-Bouldin')
    ax.set_xticks(np.arange(0, 20, step=5))
    ax2 = ax.twinx()
    ax2.plot(x,scores[1],c='b',marker=(8,2,0),ls='--',label='Calinksi-Harabasz')
    ax.set_ylabel("Silhouette and Davies-Bouldin Coefficient", color="red")
    ax2.set_ylabel("Calinski-Harabasz Coefficient", color="blue")
    ax.set_title("{} Clustering Analysis".format(method))
    fig.savefig("figures/c-score-{0}".format(method), facecolor="w", edgecolor="w",
                dpi=300)
    
    
def extract_score(method):
    
    sil = []
    ch = []
    db = []
    
    for i in method:
        sil.append(i[0])
        ch.append(i[1])
        db.append(i[2])
        
    return [sil, ch, db]
        
        
    

        
        
        
        