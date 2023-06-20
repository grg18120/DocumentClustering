import DocClust.config as config
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import Birch
from sklearn_extra.cluster import CommonNNClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from matplotlib import pyplot as plt
import time



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


def dbscan(X, n_clusters, algorithm, n_jobs):
    # Normalize vectors X
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))
    # Xnormed = X

    # nearest_neighbors = NearestNeighbors(n_neighbors = config.nn)
    # neighbors = nearest_neighbors.fit(Xnormed)
    # distances, indices = neighbors.kneighbors(Xnormed)
    # distances = np.sort(distances[:,config.nn - 1], axis = 0)

    # i = np.arange(len(distances))
    # knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    # knee2 = KneeLocator(i, distances, S=1, curve='concave', direction='increasing', interp_method='polynomial')

    # e = knee.knee_y
    # e2 = knee2.knee_y

    # tmpi = [i for i, x in enumerate(distances) if i%10 == 0]
    # tmpx = [x for i, x in enumerate(distances) if i%10 == 0]
    # fig, ax = plt.subplots()
    # ax.plot(tmpi, tmpx, linewidth=2.0)
    # ax.hlines(y = [e, (e+e2)/2.0, e2], color = 'r', xmin = 0, xmax = X.shape[0])
    # plt.show()


    # print("e = ",distances[knee.knee])
    # print("e2 = ",distances[knee2.knee])

    e = 0.5
    step = 0.4
    labels_pred = [-1]
    i = 0
    num_clusters = 1
    while(num_clusters < n_clusters and abs(step) > 0.01):

        dist_clust = set([x for x in labels_pred if x != -1])
        prev_num_clusters = len(dist_clust)

        labels_pred = DBSCAN(
            eps = e, 
            min_samples = config.nn,
            algorithm = algorithm, 
            leaf_size = 30,
            metric = 'euclidean',
            n_jobs = n_jobs
        ).fit(Xnormed).labels_

        num_minus1 = len([x for x in labels_pred if x == -1])
        num_zeroclust = len([x for x in labels_pred if x == 0])
        dist_clust = set([x for x in labels_pred if x != -1])
        num_clusters = len(dist_clust)

        if (num_minus1 == X.shape[0]):
            step = step/2
        elif (num_zeroclust == X.shape[0]):
            step = - step/2
        elif (prev_num_clusters > num_clusters):
            step = - step / 2.0

        print(f"loop:{i} eps:{e} step:{step}   conunt(-1):{num_minus1}   DistClust:{dist_clust}")
        e = e + step
        i+=1
        time.sleep(2)
    
    return labels_pred


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