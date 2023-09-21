from queue import Empty
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
#from numpy import average,flatten
import numpy as np
import pandas as pd
import functools
import time
import csv
import DocClust.config as config
import DocClust.experiments as experiments
from tqdm import tqdm 
import umap
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from matplotlib import pyplot as plt
from datasets import load_dataset
import os
from collections import Counter


# ------------------------ EMBEDDINGS - WORD VECTORS ------------------------ #
def load_models(vectorizers_strings):
    """
    Function which loads pre-trained NLP models.
    This needs to run once since all models need a few seconds to load.
    """

    spacy_model_en = spacy_model_gr = None
    sent_transformers_model = jina_model = None 
    bert_model_gr = sent_transformers_paraph_multi_model_gr = None

    # Models for english datasets
    if len([item for item in config.datasets_strings if item in config.datasets_en_strings]) > 0:
        spacy_model_en = spacy.load('en_core_web_lg')
        sent_transformers_model = SentenceTransformer(
            model_name_or_path = 'sentence-transformers/all-mpnet-base-v2',
            device = 'cpu'
        ) 
        jina_model = SentenceTransformer(
            model_name_or_path = 'jinaai/jina-embedding-l-en-v1',
            device = 'cpu'
        )

    # Models for greek datasets
    if len([item for item in config.datasets_strings if item in config.datasets_gr_strings]) > 0:
        spacy_model_gr = spacy.load('el_core_news_lg')
        bert_model_gr = SentenceTransformer(
            model_name_or_path = 'nlpaueb/bert-base-greek-uncased-v1',
            device = 'cpu'
        )
        sent_transformers_paraph_multi_model_gr =  SentenceTransformer(
        model_name_or_path = 'paraphrase-multilingual-mpnet-base-v2',
            device = 'cpu'
        )
        # bart_model_gr

    return (
        spacy_model_en, 
        spacy_model_gr, 
        sent_transformers_model, 
        jina_model, 
        bert_model_gr, 
        sent_transformers_paraph_multi_model_gr
    )


def spacy_useful_token(token):
    """
    Keep useful tokens which have 
       - Part Of Speech tag (POS): ['NOUN','PROPN','ADJ']
       - Alpha(token is word): True
       - Stop words(is, the, at, ...): False
    """
    return token.pos_ in ['NOUN','PROPN','ADJ'] and token.is_alpha and not token.is_stop and token.has_vector 


def spacy_model_embeddings(corpus, spacy_model, labels_true):
    """
    Spacy embeddings
    """
    doc_vectors = []
    doc_indx = []
    for index, text in enumerate(tqdm(corpus)):
        doc = spacy_model(text)
        if doc.has_vector:
    
            vector_list = [token.vector for token in doc if spacy_useful_token(token)]
            # vector_list to np array
            doc_vector = np.mean(vector_list, axis = 0)

            # remove nan value & zero elements vectors
            if np.any(doc_vector) and isinstance(doc_vector,np.ndarray):
                doc_vectors.append(doc_vector)
                doc_indx.append(index)

    return np.array(doc_vectors, dtype = object), [labels_true[x] for x in doc_indx]


def sent_transformers_model_embeddings(corpus, spacy_model, sent_transorfmers_model, labels_true):
    doc_vectors = []
    doc_indx = []
    for index, text in enumerate(tqdm(corpus)):

        # Take sentences(Span objects) from spacy
        doc = spacy_model(text)
        sents_spacy_span = [sent for sent in doc.sents]

        # Cut sentence in the middle if len(tokens of sentence) < transf_model.max_seq_length
        sents_spacy_str = []
        for sent in sents_spacy_span:
            if len([token for token in sent]) > sent_transorfmers_model.max_seq_length :
                middle = int(len(sent.text)/2)
                sents_spacy_str.append(sent.text[:middle])
                sents_spacy_str.append(sent.text[middle:])
            else:
                sents_spacy_str.append(sent.text)

        # Mean of sentenses vectors to create doc vector
        sent_vectors =  sent_transorfmers_model.encode(sents_spacy_str)
        doc_vector = np.mean(sent_vectors, axis = 0)

        # remove nan value & zero elements vectors
        if (np.any(doc_vector) and isinstance(doc_vector,np.ndarray)):
            doc_vectors.append(doc_vector)
            doc_indx.append(index)

    return np.array(doc_vectors, dtype = object), [labels_true[x] for x in doc_indx]


def jina_model_embeddings(corpus, jina_model, labels_true):
    doc_vectors = []
    doc_indx = []
    for index, text in enumerate(tqdm(corpus)):
        doc_vector = jina_model.encode(text)

        # remove nan value & zero elements vectors
        if np.any(doc_vector) and isinstance(doc_vector,np.ndarray):
            doc_vectors.append(doc_vector)
            doc_indx.append(index)

    return np.array(doc_vectors, dtype = object), [labels_true[x] for x in doc_indx]

def tfidf(corpus, labels_true):
    if config.datasets_language == "english":
        stop_words = "english"
    elif config.datasets_language == "greek":
        stop_words = list(config.greek_stop_words)

    vectorizer = TfidfVectorizer(
        lowercase = True,
        use_idf = True,
        norm = None,
        stop_words = stop_words, #"english"
        max_df = 0.99,
        min_df = 0.01
        #max_features = 4#5250
    )
    vectorizer_fitted = vectorizer.fit_transform(tqdm(corpus))
    #feature_names = vectorizer.get_feature_names_out()

    doc_vectors = []
    doc_indx = []
    for index, doc_vector in enumerate(vectorizer_fitted.todense()):

        # remove nan value & zero elements vectors
        if (np.any(doc_vector) and isinstance(doc_vector,np.matrix)):
            doc_vectors.append(np.squeeze(np.asarray(doc_vector)))
            doc_indx.append(index)

    return np.array(doc_vectors, dtype = object), [labels_true[x] for x in doc_indx]




# ------------------------ REDUCE DIMENSIONALITY ------------------------ #

def reduce_dim_umap(vectors):
    """
    n_neighbors: default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
    n_components: default 2, The dimension of the space to embed into.
    metric: default 'euclidean', The metric to use to compute distances in high dimensional space.
    min_dist: default 0.1, The effective minimum distance between embedded points.
    """
    reducer = umap.UMAP(
        n_neighbors = 15,
        n_components = 20, 
        metric = 'cosine',
        min_dist = 0.1,
        random_state = config.random_state
    )
    # return reducer.fit_transform(vectors)
    return np.array(reducer.fit_transform(vectors), dtype = object)


# ------------------------ PARAMETER TUNING ------------------------ #

def parameter_tuning(algo_func, algo_string, labels_true, parameter_range, step):
    if (step == 0):
        parameter_values = parameter_range
    else:
        parameter_values = np.arange(parameter_range[0], parameter_range[1], step).tolist()
        
    csv_rows = []
    for i in range(len(parameter_values)):

        print(f" -- eps:{parameter_values[i]} -- ")
        labels_pred = algo_func(parameter_values[i])
        
        csv_row = [parameter_values[i]]
        for evaluation_metric_string in config.evaluation_metrics_strings:
            score  = wrapper_args(config.evaluation_metrics_pointers().get(evaluation_metric_string),[list(labels_true), list(labels_pred)])
            csv_row.append(round(score, 4))
            print(f"{evaluation_metric_string} = {score}")
        csv_rows.append(csv_row)
        print("\n\n")

    parameters_tuning_csv("Parameter",csv_rows,"".join([algo_string, str(config.nn)]))


def parameters_tuning_csv(parameter_name, data, csv_name):
    csv_file_path = "".join([config.parameters_dir, csv_name, ".csv"])
    with open(csv_file_path, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        
        writer.writerow([parameter_name] + config.evaluation_metrics_strings)
        # Write the data to the CSV file
        for row in data:
            writer.writerow(row)


def knee_points(X, nn, draw_images):

    nearest_neighbors = NearestNeighbors(n_neighbors = nn)
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    distances = np.sort(distances[:, nn - 1], axis = 0)

    i = np.arange(len(distances))
    knee_incr_convex = KneeLocator(i, distances, S = 1, curve = 'convex', direction = 'increasing', interp_method = 'polynomial')
    knee_decr_convex = KneeLocator(i, distances, S = 1, curve = 'convex', direction = 'decreasing', interp_method = 'polynomial')
    knee_incr_concave = KneeLocator(i, distances, S = 1, curve = 'concave', direction = 'increasing', interp_method = 'polynomial')
    knee_decr_concave = KneeLocator(i, distances, S = 1, curve = 'concave', direction = 'decreasing', interp_method = 'polynomial')

    if (draw_images):
        fig = plt.figure(figsize=(5, 5))
        plt.plot(distances)
        plt.xlabel("Points")
        plt.ylabel("Distance")
        plt.savefig("Distance_curve.png", dpi = 300)

        fig = plt.figure(figsize=(5, 5))
        knee_incr_convex.plot_knee()
        plt.xlabel("Points")
        plt.ylabel("Distance")
        plt.savefig("Distance_knee_incr_convex.png", dpi=300)

        fig = plt.figure(figsize=(5, 5))
        knee_decr_convex.plot_knee()
        plt.xlabel("Points")
        plt.ylabel("Distance")
        plt.savefig("Distance_knee_decr_convex.png", dpi=300)

        fig = plt.figure(figsize=(5, 5))
        knee_incr_concave.plot_knee()
        plt.xlabel("Points")
        plt.ylabel("Distance")
        plt.savefig("Distance_knee_incr_concave.png", dpi=300)

        fig = plt.figure(figsize=(5, 5))
        knee_decr_concave.plot_knee()
        plt.xlabel("Points")
        plt.ylabel("Distance")
        plt.savefig("Distance_decr_concave.png", dpi=300)

    return (
        distances[knee_incr_convex.knee], 
        distances[knee_decr_convex.knee],
        distances[knee_incr_concave.knee],
        distances[knee_decr_concave.knee]
    )


# ------------------------ ENGLISH DATASETS ------------------------ #
from sklearn.datasets import fetch_20newsgroups

def load_dataset_20newsgroups():
    newsgroups_dataset = fetch_20newsgroups(
        subset = 'all', 
        random_state = 42,
        remove = ('headers', 'footers', 'quotes') 
    )
    return [newsgroups_dataset.data, list(newsgroups_dataset.target), len(newsgroups_dataset.target_names)]

 
def load_dataset_test():
    corpus = [
        "",
        'data science is one of the most important fields of science',
        'Game of Thrones is an amazing TV series!',
        'this is one of the best data science courses',
        'data scientists analyze data',
        'Game of Thrones is the best TV series!',
        "The car is driven on the road",
        'Game of Thrones is so great.',
        "The truck is driven on the highway",
        " "
    ]
    labels_true = [2, 0, 2, 0, 0, 2, 1, 2, 1, 1]
    n_clusters = len(set(labels_true))
    return [corpus, labels_true, n_clusters]

def load_dataset_blobs():
    corpus = ["blobs"]
    n_clusters = 3
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples = 750, centers = centers, cluster_std = 0.4, random_state = 0)
    return [corpus, labels_true, n_clusters]


def load_dataset_reuters21578():
    '''
    Reuters21578 - Subset:ModAplte - Traing + Test documents
    Choose documents with only one topic(class)
    Choose 8 (topics)classes with the most amount of documents 
    '''

    dataset = load_dataset("rjjan/reuters21578", "ModApte", split = "train+test")
    dataset_indc_one_topic = []
    topics_amount_dict = {}
    for doc_id, document in enumerate(dataset):
        topics = document["topics"]
        if (len(topics) == 1):
            dataset_indc_one_topic.append(doc_id)
            topic = topics[0]
            if topic in topics_amount_dict:
                topics_amount_dict[topic] += 1
            else:
                topics_amount_dict.update({topic: 1})

    topics_amount_list = list(topics_amount_dict.items())
    topics_amount_list_sorted = sorted(topics_amount_list, key = lambda topics_amount_list:topics_amount_list[1], reverse = True)
    eight_most_freq_classes = [x[0] for x in topics_amount_list_sorted[:8]]

    corpus, labels_true = ([], [])
    for doc_inx in dataset_indc_one_topic:
        document = dataset[doc_inx]
        topic = document["topics"][0]
        if topic in eight_most_freq_classes:
            corpus.append(document["text"])
            labels_true.append(eight_most_freq_classes.index(topic))
    n_clusters = len(set(labels_true))

    # corpus2, labels_true2 = zip(*[(dataset[doc_indx]["text"], eight_most_freq_classes.index(dataset[doc_indx]["topics"][0])) for doc_indx in dataset_indc_one_topic if dataset[doc_inx]["topics"][0] in eight_most_freq_classes])

    return [corpus, labels_true, n_clusters]



def load_dataset_trec():
    dataset = load_dataset("trec", split = "train+test")
    # Load datraset except documents from class = 0 (very few)
    corpus, labels_true  = zip(*[(x["text"], x["coarse_label"]) for x in  dataset if x["coarse_label"] != 0])
    n_clusters = len(set(labels_true))

    return [list(corpus), list(labels_true), n_clusters]


def load_dataset_webace():
    webace_path = "".join([config.local_datasets_path, "WebAce\\WebAce\\"])
    webace_files = os.listdir(webace_path)
    corpus, labels_true_str =  zip(*[(open(webace_path + x, 'r').read(), x.split(".")[0]) for x in webace_files])

    labels_distnct_values = set(labels_true_str)
    labels_true = [list(labels_distnct_values).index(x) for x in labels_true_str]
    n_clusters = len(set(labels_distnct_values))

    return [list(corpus), labels_true, n_clusters]
 

def load_dataset_pubmed4000():
    pubmed4000_path = "".join([config.local_datasets_path, "pubmed4000\\pubmed4000\\"])
    pubmed4000_files = os.listdir(pubmed4000_path)
    #corpus = [open(pubmed4000_path + file_name, 'r').read() for file_name in pubmed4000_files]
    #labels_true_str = ["".join([char for char in file_name if not char.isdigit()]).split(".")[0] for file_name in pubmed4000_files]

    corpus, labels_true_str = zip(*[(open(pubmed4000_path + file_name, 'r').read(), "".join([char for char in file_name if not char.isdigit()]).split(".")[0] ) for file_name in pubmed4000_files])
    labels_true, n_clusters = labels_str_to_int(labels_true_str)

    return [list(corpus), labels_true, n_clusters]


def load_dataset_classic4():
    classic4_path = "".join([config.local_datasets_path, "classic4\\classic4_more_than_500_bytes\\"])
    classic4_files = os.listdir(classic4_path)
    #corpus = [open(classic4_path + file_name, 'r').read() for file_name in classic4_files]
    #labels_true_str = ["".join([char for char in file_name if not char.isdigit()]).split(".")[0] for file_name in classic4_files]
    corpus, labels_true_str = zip(*[(open(classic4_path + file_name, 'r').read(), "".join([char for char in file_name if not char.isdigit()]).split(".")[0]) for file_name in classic4_files ])
    labels_true, n_clusters = labels_str_to_int(labels_true_str)
    return [list(corpus), labels_true, n_clusters]


def load_dataset_greeksum():
    """
    GreeSum dataset (valid + test splits)
    """
    greeksum_path = "".join([config.local_datasets_path, "greeksum_test_valid\\"])
    greeksum_csv_file = "greeksum_test_valid.csv"
    corpus, labels_true = zip(*[(csv_row[0], int(csv_row[1])) for csv_row in csv.reader(open(greeksum_path + greeksum_csv_file, 'r', encoding='utf-8')) if csv_row[1].isdigit()])
    return [list(corpus), list(labels_true), len(set(labels_true))]

def load_dataset_makedonia():
    """
    Makedonia dataset
    """
    makedonia_path = "".join([config.local_datasets_path, "makedonia\\"])
    makedonia_csv_file = "makedonia.csv"
    corpus, labels_true_str = zip(*[(csv_row[0], csv_row[1]) for csv_row in csv.reader(open(makedonia_path + makedonia_csv_file, 'r', encoding='utf-8'))])

    labels_count = Counter(labels_true_str)
    labels_8_most_freq = sorted(labels_count, key = labels_count.get, reverse = True)[:8]

    corpus_8, labels_true_str_8 = zip(*[(doc, labels_true_str[indx]) for indx, doc in enumerate(corpus) if labels_true_str[indx] in labels_8_most_freq])
    labels_true_8, n_clusters_8 = labels_str_to_int(labels_true_str_8)


    # corpus = corpus[1:]
    # labels_true_str = labels_true_str[1:]

    # corpus_8 = []
    # labels_true_str_8 = []
    # indx_removed = []
    # for indx, doc in enumerate(corpus):
    #     if labels_true_str[indx] in labels_8_most_freq:
    #         corpus_8.append(doc)
    #         labels_true_str_8.append(labels_true_str[indx])
    #     else:               
    #         indx_removed.append(indx)

    # dataset_string = "makedonia"
    # vectorizer_string = "greek_bart_model_embeddings"
    # for ind in indx_removed:
    #     os.remove(f"precomputed_vectors\\{dataset_string}\\{vectorizer_string}\\{ind}.pkl" )

    # for i in range(7269):
    #     print(i)
    # p = "C:\\Users\\George Georgariou\\Documents\\Visual Studio Code\\DocumentClustering\\precomputed_vectors\\makedonia\\greek_bart_model_embeddings\\"
    # files = os.listdir(p)
    # files_sorted = os.listdir(p)
    # def nn(element):
    #     return int(element.split(".")[0])
    # files_sorted.sort(key = nn)

    # for i in range(len(files_sorted)):
    #     old_name = p + files_sorted[i]
    #     new_name = p + str(i) + ".pkl"

    #     # Renaming the file
    #     os.rename(old_name, new_name)

    # return [list(corpus_8), list(labels_true_8), n_clusters_8, indx_removed]

    return [list(corpus_8), list(labels_true_8), n_clusters_8]


# ------------------------ GREEK DATASET ------------------------ #

def load_dataset_greek_legal_code():
    dataset = load_dataset("greek_legal_code", "volume", split = "train+test+validation")
    corpus, labels_true  = zip(*[(x["text"], x["label"]) for x in  dataset])
    n_clusters = len(set(labels_true))

    return [list(corpus), list(labels_true), n_clusters]






# ------------------------ HELPFUL FUNCS ------------------------ #
def wrapper(func):
    return func()


def wrapper_args(func, args): 
    """
    Pass a list of arguments(args) into function(func) 
    """
    return func(*args)


def counter(func):
    """
    Print the elapsed system time in seconds, 
    if only the debug flag is set to True.
    """
    if not config.debug:
        return func
    @functools.wraps(func)
    def wrapper_counter(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f'{func.__name__}: {end_time - start_time} secs')
        return result
    return wrapper_counter


def clean_corpus(corpus, labels_true):
    """
     Remove multiple spaces
     Remove empty documents
    """
    docs, doc_indx = zip(*[(' '.join(text.split()), index) for index, text in enumerate(corpus) if len(" ".join(text.split())) > 0])
    return list(docs), [labels_true[x] for x in doc_indx]


def accepted_vector(vector, vector_type):
    """
    Check vector all values 0
    Check proper object type
    """
    return np.any(vector) and isinstance(vector,vector_type)


def labels_str_to_int(labels_str):
    unique_labels_vals = set(labels_str)
    return [ list(unique_labels_vals).index(lab_str) for lab_str in labels_str], len(unique_labels_vals)


# def create_vector(dataset_string, corpus, vectorizer_string, labels_true_corpus, spacy_model, sent_transformers_model, jina_model):
#     if (vectorizer_string == "tfidf"): 
#         if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
#             X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
#         else:   
#             X, labels_true  = wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [labels_true_corpus])
#             experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
#     if (vectorizer_string == "spacy_model_embeddings"): 
#         if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
#             X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
#         else:
#             X, labels_true  = wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model] + [labels_true_corpus])
#             experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
#     if (vectorizer_string == "sent_transformers_model_embeddings"): 
#         if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
#             X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
#         else:
#             X, labels_true  = wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model] + [sent_transformers_model] + [labels_true_corpus])
#             experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
#     if (vectorizer_string == "jina_model_embeddings"): 
#         if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
#             X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
#         else:
#             X, labels_true  = wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [jina_model] + [labels_true_corpus])
#             experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)

#     return X, labels_true
