from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import Birch
from sklearn_extra.cluster import CommonNNClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

import numpy as np

random_state = 42

def kmeans(X, n_clusters, algorithm, init_centers):
    return KMeans(
        n_clusters = n_clusters, 
        init = init_centers, 
        algorithm = algorithm,
        n_init = 10, 
        tol = 1e-4,
        random_state = random_state
    ).fit(X).labels_


def kmedoids(X, n_clusters, method, init_centers):
    return KMedoids(
        n_clusters = n_clusters,
        metric = 'cosine',
        method = method, 
        init= init_centers, 
        max_iter = 300, 
        random_state = random_state
    ).fit(X).labels_


def agglomerative(X,n_clusters, compute_full_tree, linkage, metric):
    return AgglomerativeClustering(
        n_clusters = n_clusters, 
        metric = metric, 
        connectivity = None, 
        compute_full_tree = compute_full_tree, 
        linkage = linkage, 
        distance_threshold = None, 
        compute_distances = False 
    ).fit(X).labels_


def birch(X, n_clusters):
    return Birch(
        branching_factor = 50,
        n_clusters = n_clusters, 
        threshold = 0.5
    ).fit(X).labels_


def dbscan(X, algorithm, n_jobs):
    # Normalize vectors X
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))

    # Estimate eps distance 
    distances, indices = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(X).kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    distances_dev = np.diff(distances)
    max_val_index = np.where(distances_dev == np.amax(distances_dev))[0][0]
    e = distances[max_val_index]
    #print("e = ",e)
    #plt.plot(distances)

    return DBSCAN(
        eps = e, 
        min_samples = 5,
        algorithm = algorithm, 
        leaf_size = 30,
        metric = 'euclidean',
        n_jobs = n_jobs
    ).fit(Xnormed).labels_


def meanshift(X, bin_seeding, n_jobs):
    return MeanShift(
#        bandwidth = 1000.0,
        bin_seeding = bin_seeding,
        n_jobs = n_jobs 
    ).fit(X).labels_


def optics(X, cluster_method, algorithm, n_jobs):
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))
    return OPTICS(
		min_samples = 5,  
		metric = 'euclidean', 
		p = 1, 
		cluster_method = cluster_method, 
		algorithm = algorithm, 
		leaf_size = 30,
        n_jobs = n_jobs
	).fit(Xnormed).labels_


def common_nn(X, algorithm, n_jobs):
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))
    return CommonNNClustering(
		eps = 0.5, 
		min_samples = 5, 
		metric = 'euclidean', 
		algorithm = algorithm,
		leaf_size = 30, 
		p = 1,
        n_jobs = n_jobs
	).fit(Xnormed).labels_