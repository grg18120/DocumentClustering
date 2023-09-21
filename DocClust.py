import pandas as pd
import numpy as np
import time
import DocClust.config as config 
import DocClust.utils as utils
import DocClust.experiments as experiments
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import os
import pickle


# cor, lab_tr, n_clusters, indx_removed  = utils.load_dataset_makedonia()   
   

# Create directories if they doesnt exist to store vectors-embedding 
experiments.create_serialized_vectors_dirs()

# Load Models to create embeddings
# [spacy_model_en, spacy_model_gr, sent_transformers_model, bert_model_gr, jina_model] = utils.load_models(config.vectorizers_strings)
(
    spacy_model_en, 
    spacy_model_gr, 
    sent_transformers_model, 
    jina_model, 
    bert_model_gr, 
    sent_transformers_paraph_multi_model_gr
) = utils.load_models(config.vectorizers_strings)

# Main Loops
for dataset_string in config.datasets_strings:
    [corpus, labels_true, n_clusters]  = utils.wrapper(config.datasets_pointers().get(dataset_string))
    # experiments.plot_histogram(labels_true)


    print("Corpus Size before clean: ", len(corpus))
    corpus, labels_true = utils.clean_corpus(corpus, labels_true)
    # experiments.plot_histogram(labels_true, dataset_string)
    labels_true_corpus = labels_true[:]
    print("Corpus Size After clean: ", len(corpus))

    for vectorizer_string in config.vectorizers_strings:
        startTimeVectorizer = time.time()
        if (vectorizer_string == "tfidf"): 
            if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
            else:   
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
        if (vectorizer_string == "spacy_model_embeddings"): 
            if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
            else:
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_en] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
        if (vectorizer_string == "sent_transformers_model_embeddings"): 
            if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
            else:
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_en] + [sent_transformers_model] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
        if (vectorizer_string == "jina_model_embeddings"): 
            if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
            else:
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [jina_model] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)


        if (vectorizer_string == "greek_spacy_model_embeddings"): 
            if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
            else:
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_gr] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
        if (vectorizer_string == "greek_bert_model_embeddings"): 
            if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
            else:
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_gr] + [bert_model_gr] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
        if (vectorizer_string == "sent_transformers_paraph_multi_model_embeddings"): 
            if (experiments.check_folder_size(dataset_string, vectorizer_string) > 1000):
                X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
            else:
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_gr] + [sent_transformers_paraph_multi_model_gr] + [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)
        if (vectorizer_string == "greek_bart_model_embeddings"): 
            X, labels_true = experiments.load_deselialized_vector(dataset_string, vectorizer_string)
            


        print("\n\n*******************************************************************")
        print("*******************************************************************")
        print(f"Vectorization Method: {vectorizer_string} time = {time.time() - startTimeVectorizer}")
        print("*******************************************************************")
        print("*******************************************************************\n\n")
        
        # Remove vectors with vector = nparray([nan, nan, ......, nan]) or vector_norm = 0
        print(f"X Size = {len(X)}")
        print(f"X Shape = {X.shape}")
        print(f"labels_true Size = {len(labels_true)}")

        # Reduce dimensionality
        if (config.reduce_dim and vectorizer_string == "sent_transformers_model_embeddings"):
            X = utils.reduce_dim_umap(X)
            print("\n\nReduce dimensions UMAP")
            print(f"X Size = {len(X)}")
            print(f"X Shape = {X.shape}\n\n")

        all_eval_metric_values = []
        for clustering_algorithms_string in config.clustering_algorithms_strings:
            startTimeClustAlgo = time.time()
            arguments_list = config.clustering_algorithms_arguments(n_clusters).get(clustering_algorithms_string)
            for arguments in arguments_list:
                #labels_pred = list(utils.wrapper_args(config.clustering_algorithms_pointers().get(clustering_algorithms_string), [X] + arguments ))
                labels_pred = list(utils.wrapper_args(config.clustering_algorithms_pointers().get(clustering_algorithms_string), [X] + [labels_true] + arguments ))
                
                
                if (dataset_string == "test"): print(f"labels_true = {labels_true}")
                if (dataset_string == "test"): print(f"labels_pred = {labels_pred}")
                #print(f"labels_true = {labels_true}")
                #print(f"labels_pred = {labels_pred}")

                print(f"\n{clustering_algorithms_string}({arguments})\n")

                for evaluation_metric_string in config.evaluation_metrics_strings:
                    score  = utils.wrapper_args(config.evaluation_metrics_pointers().get(evaluation_metric_string),[list(labels_true), list(labels_pred)])
                    all_eval_metric_values.append(score)
                    print(f"{evaluation_metric_string} = {score}")

            print(f"\ntime for {clustering_algorithms_string} = {(time.time() - startTimeClustAlgo)/60}\n\n")

        print(f"\n\ntime for {vectorizer_string} = {(time.time() - startTimeVectorizer)/60}\n\n")
        experiments.save_csv(dataset_string, vectorizer_string, n_clusters, all_eval_metric_values)      
        