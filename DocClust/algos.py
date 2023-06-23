import DocClust.config as config

import DocClust.metrics as metrics  
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


def dbscan(X, labels_true, n_clusters, algorithm, n_jobs):
    # Normalize vectors X
    Xnorm = np.linalg.norm(X.astype(float), axis = 1)
    Xnormed = np.divide(X, Xnorm.reshape(Xnorm.shape[0], 1))
    # Xnormed = X

    nearest_neighbors = NearestNeighbors(n_neighbors = config.nn)
    neighbors = nearest_neighbors.fit(Xnormed)
    distances, indices = neighbors.kneighbors(Xnormed)
    distances = np.sort(distances[:,config.nn - 1], axis = 0)

    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    # knee2 = KneeLocator(i, distances, S=1, curve='concave', direction='increasing', interp_method='polynomial')

    e = knee.knee_y
    print(f"Proposal eps:{e}")
    # e2 = knee2.knee_y

    # tmpi = [i for i, x in enumerate(distances) if i%10 == 0]
    # tmpx = [x for i, x in enumerate(distances) if i%10 == 0]
    # fig, ax = plt.subplots()
    # ax.plot(tmpi, tmpx, linewidth=2.0)
    # ax.hlines(y = [e, (e+e2)/2.0, e2], color = 'r', xmin = 0, xmax = X.shape[0])
    # plt.show()


    # print("e = ",distances[knee.knee])
    # print("e2 = ",distances[knee2.knee])

    # e = 1.007
    # step = - 0.1
    # labels_pred = [-1]
    # i = 0
    # num_clusters = 1
    # # num_clusters < n_clusters and
    # while( abs(step) > 0.001):

    #     dist_clust = set([x for x in labels_pred if x != -1])
    #     prev_num_clusters = len(dist_clust)
    #     prev_num_minus1 = len([x for x in labels_pred if x == -1])

    #     labels_pred = DBSCAN(
    #         eps = e, 
    #         min_samples = config.nn,
    #         algorithm = algorithm, 
    #         leaf_size = 30,
    #         metric = 'euclidean',
    #         n_jobs = n_jobs
    #     ).fit(Xnormed).labels_

    #     num_minus1 = len([x for x in labels_pred if x == -1])
    #     num_zeroclust = len([x for x in labels_pred if x == 0])
    #     dist_clust = set([x for x in labels_pred if x != -1])
    #     num_clusters = len(dist_clust)

    #     if prev_num_clusters < num_clusters :
    #         e = e - step 
    #         step = step / 1.5
    #     if prev_num_clusters >= num_clusters and prev_num_minus1 < num_minus1 and prev_num_clusters > 1:
    #         e = e - step 
    #         step = step / 1.5

    #     print(f"loop:{i} eps:{e} step:{step}   conunt(-1):{num_minus1}   DistClust:{dist_clust}")
    #     e = e + step
    #     i+=1
    #     time.sleep(1)

    def dbs(ee):
        return DBSCAN(
            eps = ee, 
            min_samples = config.nn,
            algorithm = algorithm, 
            leaf_size = 30,
            metric = 'euclidean',
            n_jobs = n_jobs
        ).fit(Xnormed).labels_

    
    step = 0.05
    eps_values = np.arange(0.06, 1.2, step).tolist()
    eps_new_values = []
    evaluation_metrics = []
    
    
    for i in range(len(eps_values)):
        e = eps_values[i]
        labels_pred = dbs(e)
        
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
                labels_pred = dbs(ee)
                metrics_value = metrics.v_measure_index(labels_true, labels_pred)
                eps_new_values.append(ee)
                evaluation_metrics.append(metrics_value)
                print(f"eps:{ee}  metric:{metrics_value}")
                #print(f"labels_pred:{labels_true}")
                #print(f"labels_pred:{labels_pred}\n")
            
            ee = e + step/2.0
            if ee not in eps_new_values:
                labels_pred = dbs(ee)
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

    return dbs(max_eps)


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