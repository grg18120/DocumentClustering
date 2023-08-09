import DocClust.config as config
import DocClust.metrics as metrics  
import DocClust.utils as utils
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from hdbscan import hdbscan_
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
import csv
import numpy as np

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


def dbscan(X, labels_true, n_clusters, algorithm, n_jobs):
    print("-------------------------------------\n")
    algo_string = "dbscan"

    # Normalize vectors X
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))

    return DBSCAN(
        eps = 0.87, 
        min_samples = 100,
        algorithm = algorithm, 
        leaf_size = 30,
        metric = 'euclidean',
        n_jobs = n_jobs
    ).fit(Xnormed).labels_

    def dbs(ee):
        return DBSCAN(
            eps = ee, 
            min_samples = config.nn,
            algorithm = algorithm, 
            leaf_size = 30,
            metric = 'euclidean',
            n_jobs = n_jobs
        ).fit(Xnormed).labels_

    utils.parameter_tuning(dbs, algo_string, labels_true, [0.77, 0.911], 0.01)

    max_eps = 1 
    return dbs(max_eps)


def hdbscan(X, labels_true, n_clusters, cluster_selection_method):
    algo_string = "hdbscan"

    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))



    # return hdbscan_.HDBSCAN(
    #     cluster_selection_method = cluster_selection_method
    # ).fit(Xnormed).labels_

    def hdb(min_sampless):
        return hdbscan_.HDBSCAN(
            min_samples = min_sampless,
            cluster_selection_epsilon = 0.75
        ).fit(Xnormed).labels_

    utils.parameter_tuning(hdb, algo_string, labels_true, [2, 10, 25, 50, 75, 100, 150, 200], 0)


    return hdb(10)


def meanshift(X, labels_true, n_clusters, bin_seeding, n_jobs):
    print("-------------------------------------\n")
    algo_string = "meanshift"

    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))

    return MeanShift(
        bandwidth = 0.75,
        bin_seeding = bin_seeding,
        n_jobs = n_jobs 
    ).fit(Xnormed).labels_


    def ms(bb):
        return MeanShift(
            bandwidth = bb,
            bin_seeding = False,
            n_jobs = n_jobs 
        ).fit(Xnormed).labels_

    utils.parameter_tuning(ms, algo_string, labels_true, [0.1, 1.0], 0.05)

    max_b = 0.5
    return ms(max_b)


def optics(X, labels_true, n_clusters, cluster_method, algorithm, n_jobs):
    algo_string = "optics"
    print("-------------------------------------\n")
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))

    return OPTICS(
        min_samples = 3,  
        #metric = 'euclidean', 
        p = 2, 
        cluster_method = cluster_method, 
        algorithm = algorithm, 
        leaf_size = 30,
        n_jobs = n_jobs
    ).fit(Xnormed).labels_

    def optc(eee):
        return OPTICS(
            min_samples = config.nn,  
            eps = eee,
            #metric = 'euclidean', 
            p = 2, 
            cluster_method = cluster_method, 
            algorithm = algorithm, 
            leaf_size = 30,
            n_jobs = n_jobs
        ).fit(Xnormed).labels_
    
    utils.parameter_tuning(optc, algo_string, labels_true, [0.1, 1.1], 0.1)
      
    return optc(5)
    



def common_nn(X, labels_true, n_clusters, algorithm, n_jobs):
    algo_string = "common_nn"

    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))

    return CommonNNClustering(
		eps = 0.79, 
		min_samples = 2, 
		metric = 'euclidean', 
		algorithm = algorithm,
		leaf_size = 30, 
		p = 1,
        n_jobs = n_jobs
	).fit(Xnormed).labels_

    # Parameters Tuning 
    def cnn(ee):
        return CommonNNClustering(
            eps = ee, 
            min_samples = config.nn, 
            #metric = 'euclidean', 
            algorithm = algorithm,
            leaf_size = 30, 
            p = 2,
            n_jobs = n_jobs
        ).fit(Xnormed).labels_
    
    utils.parameter_tuning(cnn, algo_string, labels_true, [0.91, 1.1], 0.01)

    max_eps = 1.0
    return cnn(max_eps)