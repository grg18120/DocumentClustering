import DocClust.algos as algos
import DocClust.metrics as metrics
import DocClust.utils as utils



# Config values. 
csv_dir = 'C:/Users/George Georgariou/Desktop/'
figures_dir = 'C:/Users/George Georgariou/Documents/Visual Studio Code/DocumentClustering/figures/'
debug = False
reduce_dim = False
nn = 75
n_jobs = None
test_dataset = False
limit_corpus_size = 0

# ------------------------ Datasets - Corpus ------------------------ #

datasets_strings = [
    "20newsgroups"
    #"test"
    #"blobs"
]

def datasets_pointers():
    return {
        "20newsgroups": utils.load_dataset_20newsgroups,
        "test": utils.load_dataset_test,
        "blobs":utils.load_dataset_blobs
    }

# ------------------------ Embeddings - Doc Vectors ------------------------ #
vectorizers_strings = [
    #"tfidf",
    #"spacy_model_embeddings",
    "sent_transformers_model_embeddings"
    
]

def vectorizers_pointers():
    return {
        "sent_transformers_model_embeddings": utils.sent_transformers_model_embeddings,
        "spacy_model_embeddings": utils.spacy_model_embeddings,
        "tfidf": utils.tfidf
    }


# ------------------------ Clustering Algorithms ------------------------ #
clustering_algorithms_strings = [
     #"kmeans",
     #"kmedoids",
     #"agglomerative",
     #"birch",
     "dbscan"
     #"meanshift",
     #"optics",
     #"meanshift",
     #"common_nn"
]

# Config Clustering algorithm approaches
def clustering_algorithms_parameteres():
    return {
        "kmeans": 
            ['n_clusters', 'algorithm', 'init_centers'],
        "kmedoids":
            ['n_clusters', 'method', 'init_centers'],
        "agglomerative": 
            ['n_clusters', 'compute_full_tree', 'linkage', 'metric'],
        "birch":
            ['n_clusters'],
        "dbscan":
            ['n_clusters', 'algorithm', 'n_jobs'],
        "meanshift": 
            ['n_clusters', 'bin_seeding', 'n_jobs'],
        "optics":
            ['n_clusters','cluster_method', 'algorithm', 'n_jobs'],
        "common_nn":
            ['n_clusters', 'algorithm', 'n_jobs']
    }

def clustering_algorithms_arguments(n_clusters):
    return {
        "kmeans": [
            [n_clusters, 'elkan', 'random'],
            [n_clusters, 'lloyd', 'random'],
            [n_clusters, 'elkan', 'k-means++'],
            [n_clusters, 'lloyd', 'k-means++']
        ],
        "kmedoids":[
            [n_clusters, 'pam', 'build'],
            [n_clusters, 'pam', 'k-medoids++'],
            [n_clusters, 'alternate', 'build'],
            [n_clusters, 'alternate', 'k-medoids++']
        ],
        "agglomerative": [
            [n_clusters, True, 'ward', 'euclidean'],
            [n_clusters, True, 'single', 'cosine'],
            [n_clusters, True, 'average', 'cosine'],
            [n_clusters, True, 'complete', 'cosine'],
            [n_clusters, False, 'ward', 'euclidean'],
            [n_clusters, False, 'single', 'cosine'],
            [n_clusters, False, 'average', 'cosine'],
            [n_clusters, False, 'complete', 'cosine']
        ],
        "birch":[
            [n_clusters]
        ],
        "dbscan":[
            [n_clusters, 'kd_tree', n_jobs],
            [n_clusters, 'ball_tree', n_jobs]
        ],
        "meanshift": [
            [n_clusters, False, n_jobs],
            [n_clusters, True, n_jobs]
        ],
        "optics":[
            [n_clusters, 'xi', 'kd_tree', n_jobs],
            [n_clusters, 'xi', 'ball_tree', n_jobs],
            [n_clusters, 'dbscan', 'kd_tree', n_jobs],
            [n_clusters, 'dbscan', 'kd_tree', n_jobs]
        ],
        "common_nn":[
            [n_clusters, 'kd_tree', n_jobs],
            [n_clusters, 'ball_tree', n_jobs]
        ]
    }
        
def clustering_algorithms_pointers():
    return {
        "kmeans": algos.kmeans,
        "kmedoids": algos.kmedoids,
        "agglomerative": algos.agglomerative,
        "birch": algos.birch,
        "dbscan": algos.dbscan,
        "meanshift": algos.meanshift,
        "optics": algos.optics,
        "common_nn": algos.common_nn
    }



# ------------------------ Ext Evaluation Metrics ------------------------ #
evaluation_metrics_strings = [
    "accuracy",
    "mutual_information",
    "adjusted_mutual_information",
    "rand_index",
    "adjusted_rand_index",
    "fowlkes_mallows_index",
    "v_measure_index",
    "homogenity",
    "completeness"
]

def evaluation_metrics_pointers():
    return {
        "accuracy": metrics.accuracy,
        "mutual_information": metrics.mutual_information,
        "adjusted_mutual_information": metrics.adjusted_mutual_information,
        "rand_index": metrics.rand_index,
        "adjusted_rand_index": metrics.adjusted_rand_index,
        "fowlkes_mallows_index": metrics.fowlkes_mallows_index,
        "v_measure_index": metrics.v_measure_index,
        "homogenity": metrics.homogenity,
        "completeness": metrics.completeness
    }
