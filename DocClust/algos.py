import DocClust.config as config
import DocClust.metrics as metrics  
import DocClust.utils as utils
from sklearn.cluster import KMeans
from hdbscan import hdbscan_
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import Birch
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import time
import csv
import numpy as np
from sklearn.metrics import silhouette_score
from collections import Counter

random_state = 42

def parameters_csv(parameter_name, data, csv_name):
    csv_file_path = "".join([config.parameters_dir, csv_name, ".csv"])
    with open(csv_file_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        
        writer.writerow([parameter_name] + config.evaluation_metrics_strings)
        # Write the data to the CSV file
        for row in data:
            writer.writerow(row)




def kmeans(X, labels_true, n_clusters, algorithm, init_centers):
    return KMeans(
        n_clusters = n_clusters, 
        init = init_centers, 
        algorithm = algorithm,
        n_init = 10, 
        tol = 1e-4,
        random_state = random_state
    ).fit(X).labels_


def kmedoids(X, labels_true, n_clusters, method, init_centers):
    return KMedoids(
        n_clusters = n_clusters,
        metric = 'cosine',
        method = method, 
        init= init_centers, 
        max_iter = 300, 
        random_state = random_state
    ).fit(X).labels_


def agglomerative(X, labels_true, n_clusters, compute_full_tree, linkage, metric):
    return AgglomerativeClustering(
        n_clusters = n_clusters, 
        metric = metric, 
        connectivity = None, 
        compute_full_tree = compute_full_tree, 
        linkage = linkage, 
        distance_threshold = None, 
        compute_distances = False 
    ).fit(X).labels_


def birch(X, labels_true, n_clusters):
    return Birch(
        branching_factor = 50,
        n_clusters = n_clusters, 
        threshold = 0.5
    ).fit(X).labels_


def hdbscan(X, labels_true, n_clusters, cluster_selection_method):
    algo_string = "hdbscan"

    # Normalize vectors X
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))

    return hdbscan_.HDBSCAN(
        min_samples = 25,
        cluster_selection_epsilon = 0.805,
        cluster_selection_method = cluster_selection_method
    ).fit(Xnormed).labels_

    def hdb(ee):
        return hdbscan_.HDBSCAN(
            min_samples = 25,
            cluster_selection_epsilon = ee
        ).fit(Xnormed).labels_

    utils.parameter_tuning(hdb, algo_string, labels_true, [0.7, 0.902], 0.005)


    return hdb(10)

