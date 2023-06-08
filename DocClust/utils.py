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
            doc_vector = np.mean(vector_list, axis=0)

            # remove nan value & zero elements vectors
            if np.any(doc_vector) and isinstance(doc_vector,np.ndarray):
                doc_vectors.append(doc_vector)
                doc_indx.append(index)

    return np.array(doc_vectors, dtype = object) , [labels_true[x] for x in doc_indx]


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


# ------------------------ ENGLISH DATASETS ------------------------ #
from sklearn.datasets import fetch_20newsgroups

def load_dataset_20newsgroups():
    newsgroups_dataset = fetch_20newsgroups(
        subset = 'all', 
        random_state = 42,
        remove = ('headers', 'footers', 'quotes') 
    )
    return [newsgroups_dataset.data, list(newsgroups_dataset.target), len(newsgroups_dataset.target_names)]

 
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
