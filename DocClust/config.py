import DocClust.algos as algos
import DocClust.metrics as metrics
import DocClust.utils as utils



# Config values. 
csv_dir = 'C:/Users/George Georgariou/Desktop/'
figures_dir = 'C:/Users/George Georgariou/Documents/Visual Studio Code/DocumentClustering/figures/'
parameters_dir = 'C:/Users/George Georgariou/Desktop/'
local_datasets_path = 'D:\\Datasets\\'
debug = False
reduce_dim = False
nn = 28
min_cluster_size = 20
n_jobs = None
test_dataset = False
limit_corpus_size = 0
random_state = 42

# ------------------------ Datasets - Corpus ------------------------ #

datasets_language = "greek"

datasets_strings = [
    #"20newsgroups"
    #"test"
    #"blobs"
    #"reuters21578",
    #"trec",
    #"webace"
    #"pubmed4000",
    #"classic4"
    # "greek_legal_code"
    "makedonia",
    "greeksum",
    "greek_legal_code"

]

def datasets_pointers():
    return {
        "20newsgroups": utils.load_dataset_20newsgroups,
        "reuters21578": utils.load_dataset_reuters21578,
        "trec": utils.load_dataset_trec,
        "classic4": utils.load_dataset_classic4,
        "pubmed4000":utils.load_dataset_pubmed4000,
        "webace": utils.load_dataset_webace,
        "greek_legal_code": utils.load_dataset_greek_legal_code,
        "test": utils.load_dataset_test,
        "blobs": utils.load_dataset_blobs,
        "makedonia": utils.load_dataset_makedonia,
        "greeksum": utils.load_dataset_greeksum,
    }

# ------------------------ Embeddings - Doc Vectors ------------------------ #
vectorizers_strings = [
    "tfidf",
    "spacy_model_embeddings",
    "sent_transformers_model_embeddings",
    #"jina_model_embeddings"   
]

def vectorizers_pointers():
    return {
        "sent_transformers_model_embeddings": utils.sent_transformers_model_embeddings,
        "jina_model_embeddings": utils.jina_model_embeddings,
        "spacy_model_embeddings": utils.spacy_model_embeddings,
        "tfidf": utils.tfidf
    }


# ------------------------ Clustering Algorithms ------------------------ #
clustering_algorithms_strings = [
     "kmeans",
     "kmedoids",
     "agglomerative",
     "birch",
     "hdbscan"
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
        "hdbscan":
            ['n_clusters', 'cluster_selection_method']
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
            [n_clusters, True, 'complete', 'cosine'],
            [n_clusters, False, 'ward', 'euclidean'],
            [n_clusters, False, 'complete', 'cosine']
        ],
        "birch":[
            [n_clusters]
        ],
        "hdbscan":[
            #[n_clusters, 'eom'],
            [n_clusters, 'leaf']
        ]
    }
        
def clustering_algorithms_pointers():
    return {
        "kmeans": algos.kmeans,
        "kmedoids": algos.kmedoids,
        "agglomerative": algos.agglomerative,
        "birch": algos.birch,
        "hdbscan": algos.hdbscan
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
