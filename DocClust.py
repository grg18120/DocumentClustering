import pandas as pd
import numpy as np
import time
import DocClust.config as config 
import DocClust.utils as utils
import DocClust.experiments as experiments
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs


# Create directories if they doesnt exist to store vectors-embedding 
experiments.create_serialized_vectors_dirs()

# Load Models to create embeddings
[spacy_model_en, spacy_model_gr, sent_transorfmers_model] = utils.load_models(config.vectorizers_strings)

# Main Loops
for dataset_string in config.datasets_strings:
    [corpus, labels_true, n_clusters]  = utils.wrapper(config.datasets_pointers().get(dataset_string))
   
    # Limit size of corpus
    if (config.limit_corpus_size>1):
        corpus_size = len(corpus)
        test_sub_corpus = int(corpus_size/config.limit_corpus_size)
        corpus = corpus[0:test_sub_corpus]
        labels_true = labels_true[0:test_sub_corpus]
        n_clusters = len(set(labels_true))
        corpus_size = len(corpus)

    print("Corpus Size before clean: ",len(corpus))
    corpus, labels_true = utils.clean_corpus(corpus, labels_true)
    labels_true_corpus = labels_true[:]
    print("Corpus Size After clean: ",len(corpus))

    # sss = set(labels_true)
    # axx = plt.hist(labels_true, density=False, bins=list(sss))   
    # print(axx)
    

    for vectorizer_string in config.vectorizers_strings:
        startTimeVectorizer = time.time()
        if (vectorizer_string == "tfidf"): 
            ssssss = experiments.check_folder_size(dataset_string, vectorizer_string)
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
                X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_en] +[sent_transorfmers_model]+ [labels_true_corpus])
                experiments.store_serialized_vector(dataset_string, vectorizer_string, X, labels_true)

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
            print("reduce dimensions UMAP")

        n_clusters = 3
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, labels_true,  = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

        all_eval_metric_values = []
        for clustering_algorithms_string in config.clustering_algorithms_strings:
            startTimeClustAlgo = time.time()
            arguments_list = config.clustering_algorithms_arguments(n_clusters).get(clustering_algorithms_string)
            for arguments in arguments_list:
                labels_pred = list(utils.wrapper_args(config.clustering_algorithms_pointers().get(clustering_algorithms_string), [X] + arguments))
                
                print("-------------------------------------\n")
                if (dataset_string == "test"): print(f"labels_true = {labels_true}")
                if (dataset_string == "test"): print(f"labels_pred = {labels_pred}")
                #print(f"labels_true = {labels_true}")
                #print(f"labels_pred = {labels_pred}")

                if (config.reduce_dim and vectorizer_string == "sent_transformers_model_embeddings" and dataset_string == "test"):
                    experiments.plotClustResults(X, labels_pred, dataset_string, clustering_algorithms_string)

                print(f"{clustering_algorithms_string}({arguments})")

                for evaluation_metric_string in config.evaluation_metrics_strings:
                    score  = utils.wrapper_args(config.evaluation_metrics_pointers().get(evaluation_metric_string),[list(labels_true), list(labels_pred)])
                    all_eval_metric_values.append(score)
                    print(f"{evaluation_metric_string} = {score}")

            print(f"\ntime for {clustering_algorithms_string} = {(time.time() - startTimeClustAlgo)/60}\n\n")

        print(f"\n\ntime for {vectorizer_string} = {(time.time() - startTimeVectorizer)/60}\n\n")
        experiments.save_csv(dataset_string, vectorizer_string, n_clusters, all_eval_metric_values)      
        