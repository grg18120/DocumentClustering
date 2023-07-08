import DocClust.config as config
import DocClust.metrics as metrics  
import DocClust.utils as utils
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

    # Normalize vectors X
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))

    # return DBSCAN(
    #     eps = 0.5, 
    #     min_samples = config.nn,
    #     algorithm = algorithm, 
    #     leaf_size = 30,
    #     metric = 'euclidean',
    #     n_jobs = n_jobs
    # ).fit(Xnormed).labels_

    def dbs(ee):
        return DBSCAN(
            eps = ee, 
            min_samples = config.nn,
            algorithm = algorithm, 
            leaf_size = 30,
            metric = 'euclidean',
            n_jobs = n_jobs
        ).fit(Xnormed).labels_

    
    step = 0.01
    eps_values = np.arange(0.6, 0.9, step).tolist()
    # eps_new_values = []
    # evaluation_metrics = []
    
    
    for i in range(len(eps_values)):
        e = eps_values[i]
        labels_pred = dbs(e)
        
        # metrics_value = metrics.v_measure_index(labels_true, labels_pred)
        # eps_new_values.append(e)
        print(f" -- eps:{e} -- ")

        #print(f"eps:{e}  metric:{metrics_value}")
        #print(f"labels_pred:{labels_true}")
        #print(f"labels_pred:{labels_pred}\n")

        # tmp = [x for x in evaluation_metrics]
        # if (len(tmp) == len(evaluation_metrics) and len(tmp)!=0):
        #     evaluation_metrics.append(metrics_value)

        #     ee = e - step/2.0
        #     if ee not in eps_new_values:
        #         labels_pred = dbs(ee)
        #         metrics_value = metrics.v_measure_index(labels_true, labels_pred)
        #         eps_new_values.append(ee)
        #         evaluation_metrics.append(metrics_value)
        #         print(f"eps:{ee}  metric:{metrics_value}")
        #         #print(f"labels_pred:{labels_true}")
        #         #print(f"labels_pred:{labels_pred}\n")
            
        #     ee = e + step/2.0
        #     if ee not in eps_new_values:
        #         labels_pred = dbs(ee)
        #         metrics_value = metrics.v_measure_index(labels_true, labels_pred)
        #         eps_new_values.append(ee)
        #         evaluation_metrics.append(metrics_value)
        #         print(f"eps:{ee}  metric:{metrics_value} ")
        #         #print(f"labels_pred:{labels_true}")
        #         #print(f"labels_pred:{labels_pred}\n")
        # else:
        #     evaluation_metrics.append(metrics_value)


        for evaluation_metric_string in config.evaluation_metrics_strings:
            score  = utils.wrapper_args(config.evaluation_metrics_pointers().get(evaluation_metric_string),[list(labels_true), list(labels_pred)])
            print(f"{evaluation_metric_string} = {score}")

        print("\n\n")

    # evaluation_metrics.append(metrics_value)
    # print("\nBest Value")
    # max_eval_metric = max(evaluation_metrics)
    # max_index = evaluation_metrics.index(max_eval_metric)
    # max_eps = eps_new_values[max_index]
    # print(f"eps:{max_eps} metric:{max_eval_metric}    ")

    max_eps = 1

    return dbs(max_eps)


def meanshift(X, labels_true, n_clusters, bin_seeding, n_jobs):
    print("-------------------------------------\n")

    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))

    return MeanShift(
        #bandwidth = bb,
        bin_seeding = bin_seeding,
        n_jobs = n_jobs 
    ).fit(Xnormed).labels_


    def ms(bb):
        return MeanShift(
            bandwidth = bb,
            bin_seeding = False,
            n_jobs = n_jobs 
        ).fit(X).labels_

    
    step = 0.05
    b_values = np.arange(0.8, 1.2, step).tolist()
    b_new_values = []
    evaluation_metrics = []
    
    
    for i in range(len(b_values)):
        b = b_values[i]
        labels_pred = ms(b)
        
        metrics_value = metrics.v_measure_index(labels_true, labels_pred)
        b_new_values.append(b)
        print(f"bandwidth:{b}  metric:{metrics_value}")
        #print(f"labels_pred:{labels_true}")
        #print(f"labels_pred:{labels_pred}\n")

        tmp = [x for x in evaluation_metrics]
        if (len(tmp) == len(evaluation_metrics) and len(tmp)!=0):
            evaluation_metrics.append(metrics_value)

            bb = b - step/2.0
            if bb not in b_new_values:
                labels_pred = ms(bb)
                metrics_value = metrics.v_measure_index(labels_true, labels_pred)
                b_new_values.append(bb)
                evaluation_metrics.append(metrics_value)
                print(f"bandwidth:{bb}  metric:{metrics_value}")
                #print(f"labels_pred:{labels_true}")
                #print(f"labels_pred:{labels_pred}\n")
            
            bb = b + step/2.0
            if bb not in b_new_values:
                labels_pred = ms(bb)
                metrics_value = metrics.v_measure_index(labels_true, labels_pred)
                b_new_values.append(bb)
                evaluation_metrics.append(metrics_value)
                print(f"bandwidth:{bb}  metric:{metrics_value} ")
                #print(f"labels_pred:{labels_true}")
                #print(f"labels_pred:{labels_pred}\n")
        else:
            evaluation_metrics.append(metrics_value)

    print("\nBest Value")
    max_eval_metric = max(evaluation_metrics)
    max_index = evaluation_metrics.index(max_eval_metric)
    max_b = b_new_values[max_index]
    print(f"bandwidth:{max_b} metric:{max_eval_metric}    ")

    return ms(max_b)


def optics(X, labels_true, n_clusters, cluster_method, algorithm, n_jobs):
    print("-------------------------------------\n")
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))

    return OPTICS(
        min_samples = config.nn,  
        #metric = 'euclidean', 
        p = 2, 
        cluster_method = cluster_method, 
        algorithm = algorithm, 
        leaf_size = 30,
        n_jobs = n_jobs
    ).fit(Xnormed).labels_

    def optc(n_n):
        return OPTICS(
            min_samples = n_n,  
            #metric = 'euclidean', 
            p = 2, 
            cluster_method = cluster_method, 
            algorithm = algorithm, 
            leaf_size = 30,
            n_jobs = n_jobs
        ).fit(Xnormed).labels_
    
    step = 100
    nn_values = np.arange(900, 1000, step).tolist()
    nn_new_values = []
    evaluation_metrics = []

    for i in range(len(nn_values)):
        nn = nn_values[i]
        labels_pred = optc(int(nn))
        
        metrics_value = metrics.v_measure_index(labels_true, labels_pred)
        nn_new_values.append(nn)
        print(f"nn:{nn}  metric:{metrics_value}")
        #print(f"labels_pred:{labels_true}")
        #print(f"labels_pred:{labels_pred}\n")

        tmp = [x for x in evaluation_metrics if metrics_value - x >= 0.05]
        if (len(tmp) == len(evaluation_metrics) and len(tmp)!=0):
            evaluation_metrics.append(metrics_value)

            nn_n = nn - step/2.0
            if nn_n not in nn_new_values:
                labels_pred = optc(int(nn_n))
                metrics_value = metrics.v_measure_index(labels_true, labels_pred)
                nn_new_values.append(nn_n)
                evaluation_metrics.append(metrics_value)
                print(f"nn:{nn_n}  metric:{metrics_value}")
                #print(f"labels_pred:{labels_true}")
                #print(f"labels_pred:{labels_pred}\n")
            
            nn_n = nn + step/2.0
            if nn_n not in nn_new_values:
                labels_pred = optc(int(nn_n))
                metrics_value = metrics.v_measure_index(labels_true, labels_pred)
                nn_new_values.append(nn_n)
                evaluation_metrics.append(metrics_value)
                print(f"nn:{nn_n}  metric:{metrics_value} ")
                #print(f"labels_pred:{labels_true}")
                #print(f"labels_pred:{labels_pred}\n")
        else:
            evaluation_metrics.append(metrics_value)

    print("\nBest Value")
    max_eval_metric = max(evaluation_metrics)
    max_index = evaluation_metrics.index(max_eval_metric)
    max_nn = nn_new_values[max_index]
    print(f"nn:{max_nn} metric:{max_eval_metric}    ")

    return optc(int(max_nn))
    



def common_nn(X, labels_true, n_clusters, algorithm, n_jobs):
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

    
    step = 0.03
    eps_values = np.arange(0.4, 1.5, step).tolist()
    eps_new_values = []
    evaluation_metrics = []
    
    
    for i in range(len(eps_values)):
        e = eps_values[i]
        labels_pred = cnn(e)
        
        metrics_value = metrics.v_measure_index(labels_true, labels_pred)
        eps_new_values.append(e)
        print(f"eps:{e}  metric:{metrics_value}")
        #print(f"labels_pred:{labels_true}")
        #print(f"labels_pred:{labels_pred}\n")

        tmp = [x for x in evaluation_metrics if metrics_value - x >= 0.05]
        if (len(tmp) == len(evaluation_metrics) and len(tmp)!=0):
            evaluation_metrics.append(metrics_value)

            ee = e - step/2.0
            if ee not in eps_new_values:
                labels_pred = cnn(ee)
                metrics_value = metrics.v_measure_index(labels_true, labels_pred)
                eps_new_values.append(ee)
                evaluation_metrics.append(metrics_value)
                print(f"eps:{ee}  metric:{metrics_value}")
                #print(f"labels_pred:{labels_true}")
                #print(f"labels_pred:{labels_pred}\n")
            
            ee = e + step/2.0
            if ee not in eps_new_values:
                labels_pred = cnn(ee)
                metrics_value = metrics.v_measure_index(labels_true, labels_pred)
                eps_new_values.append(ee)
                evaluation_metrics.append(metrics_value)
                print(f"eps:{ee}  metric:{metrics_value} ")
                #print(f"labels_pred:{labels_true}")
                #print(f"labels_pred:{labels_pred}\n")
        else:
            evaluation_metrics.append(metrics_value)

    print("\nBest Value")
    max_eval_metric = max(evaluation_metrics)
    max_index = evaluation_metrics.index(max_eval_metric)
    max_eps = eps_new_values[max_index]
    print(f"eps:{max_eps} metric:{max_eval_metric}    ")

    return cnn(max_eps)