import pandas as pd
import numpy as np
import time
import DocClust.config as config 
import DocClust.utils as utils
import DocClust.experiments as experiments


# Load Models to create embeddings
[spacy_model_en, spacy_model_gr, sent_transorfmers_model] = utils.load_models(config.vectorizers_strings)

# Main Loops
for dataset_string in config.datasets_strings:

    # Test corpus
    if (config.test_dataset):
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
        #labels_true = [0, 2, 0, 2, 2, 0, 1, 0, 1, 1]
        labels_true = [2, 0, 2, 0, 0, 2, 1, 2, 1, 1]
        n_clusters = len(set(labels_true))
    else:
        [corpus, labels_true, n_clusters]  = utils.wrapper(config.datasets_pointers().get(dataset_string))
         



    # Limit size of corpus
    if (not config.test_dataset and config.limit_corpus_size>1):
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
    

    # Remove empty text documents
    '''
    print(f"Corpus Size = {len(corpus)}")
    print(f"labels_true_corpus  = {len(labels_true)}")
    corpus, labels_true, n_clusters = remove_empty_documents(corpus, labels_true)
    labels_true_corpus = labels_true[:]
    print(f"Corpus Size afte remove empty = {len(corpus)}")
    print(f"labels_true_corpus Size afte remove empty = {len(labels_true_corpus)}")
    '''
    
    for vectorizer_string in config.vectorizers_strings:
        startTimeVectorizer = time.time()
        if (vectorizer_string == "tfidf"): X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [labels_true_corpus])
        if (vectorizer_string == "spacy_model_embeddings"): X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_en] + [labels_true_corpus])
        if (vectorizer_string == "sent_transformers_model_embeddings"): X, labels_true  = utils.wrapper_args(config.vectorizers_pointers().get(vectorizer_string), [corpus] + [spacy_model_en] +[sent_transorfmers_model]+ [labels_true_corpus])
        print("\n\n*******************************************************************")
        print("*******************************************************************")
        print(f"Vectorization Method: {vectorizer_string}")
        print("*******************************************************************")
        print("*******************************************************************\n\n")

        
        # Remove vectors with vector = nparray([nan, nan, ......, nan]) or vector_norm = 0
        print(f"X Size = {len(X)}")
        print(f"labels_true Size = {len(labels_true)}")
        '''
        #labels_true = labels_true_corpus[:]
        print(f"X Size = {len(X)}")
        print(f"labels_true Size = {len(labels_true)}")
        X, labels_true, n_clusters = remove_zeronorm_nan_vectors(X, labels_true, vectorizer_string)
        print(f"X Size afte remove norm0 and NAN values = {len(X)}")
        print(f"labels_true after remove norm0 and NAN values = {len(labels_true)}")
        '''

        # Reduce dimensionality
        if (config.reduce_dim and vectorizer_string == "sent_transformers_model_embeddings"):
            X = utils.reduce_dim_umap(X)

        all_eval_metric_values = []
        for clustering_algorithms_string in config.clustering_algorithms_strings:
            startTimeClustAlgo = time.time()
            arguments_list = config.clustering_algorithms_arguments(n_clusters).get(clustering_algorithms_string)
            for arguments in arguments_list:
                labels_pred = list(utils.wrapper_args(config.clustering_algorithms_pointers().get(clustering_algorithms_string), [X] + arguments))
                print("-------------------------------------\n")
                if (config.test_dataset): print(f"labels_true = {labels_true}")
                if (config.test_dataset): print(f"labels_pred = {labels_pred}")
                print(f"{clustering_algorithms_string}({arguments})")

                for evaluation_metric_string in config.evaluation_metrics_strings:
                    score  = utils.wrapper_args(config.evaluation_metrics_pointers().get(evaluation_metric_string),[list(labels_true), list(labels_pred)])
                    all_eval_metric_values.append(score)
                    print(f"{evaluation_metric_string} = {score}")

            print(f"\ntime for {clustering_algorithms_string} = {(time.time() - startTimeClustAlgo)/60}\n\n")

        print(f"\n\ntime for {vectorizer_string} = {(time.time() - startTimeVectorizer)/60}\n\n")
        experiments.save_csv(dataset_string, vectorizer_string, n_clusters, all_eval_metric_values)      
        