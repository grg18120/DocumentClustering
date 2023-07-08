from queue import Empty
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
#from numpy import average,flatten
import numpy as np
import pandas as pd
import functools
import time
#import DocClust.config
import DocClust.config as config
from tqdm import tqdm 
import umap
from sklearn.datasets import make_blobs


# ------------------------ EMBEDDINGS - WORD VECTORS ------------------------ #
def load_models(vectorizers_strings):
    """
    Function which loads pre-trained NLP models.
    This needs to run once since all models need a few seconds to load.
    """
    spacy_model_en = None
    spacy_model_gr = None
    sent_transformers_model = None

    if "spacy_model_embeddings" or "sent_transformers_model_embeddings" in vectorizers_strings:
        spacy_model_en = spacy.load('en_core_web_lg')
        spacy_model_gr = spacy.load('el_core_news_lg')
        if "sent_transformers_model_embeddings" in vectorizers_strings:
            sent_transformers_model = SentenceTransformer(
                model_name_or_path = 'sentence-transformers/all-mpnet-base-v2',
                device = 'cpu'
            )

    return [spacy_model_en, spacy_model_gr, sent_transformers_model]


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

        # Cut sentence in the miidle if len(tokens of sentence) < transf_model.max_seq_length
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


def tfidf(corpus, labels_true):
    vectorizer = TfidfVectorizer(
        lowercase = True,
        use_idf = True,
        norm = None,
        stop_words = "english",
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
        min_dist = 0.1
    )
    return reducer.fit_transform(vectors)


def parameter_tuning():
    pass


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
 
'''

# ------------------------ GREEK DATASET ------------------------ #

from datasets import load_dataset

dataset = load_dataset("greek_legal_code")

'''

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
    return docs, [labels_true[x] for x in doc_indx]


def accepted_vector(vector, vector_type):
    """
    Check vector all values 0
    Check proper object type
    """
    return np.any(vector) and isinstance(vector,vector_type)
