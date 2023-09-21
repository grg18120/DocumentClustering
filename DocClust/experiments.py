from functools import reduce
import csv
import os
import pickle
import pandas as pd
import numpy as np
# from DocClust.config import * 
import DocClust.config as config
from scipy.io import arff, loadmat
# import matplotlib
# matplotlib.pyplot.ion()
from matplotlib import pyplot as plt
from collections import Counter


def save_csv(dataset_name, vectorizer, n_clusters, all_eval_metric_values):
    start = 0
    approaches_count = 0
    csv_scores = {}
    approachesList = []
    evaluation_metrics = []
    for clustering_algorithms_string in config.clustering_algorithms_strings:
        argumentsList = config.clustering_algorithms_arguments(n_clusters).get(clustering_algorithms_string)
        parameters = config.clustering_algorithms_parameteres().get(clustering_algorithms_string)
        for arguments in argumentsList:
            metrics_per_approach = all_eval_metric_values[start : start + len(config.evaluation_metrics_strings)]
            evaluation_metrics.append(metrics_per_approach)
            approachesList.append(clust_algo_to_csv(clustering_algorithms_string, parameters, arguments))
            start += len(config.evaluation_metrics_strings) 
            approaches_count += 1

    array = np.zeros((len(config.evaluation_metrics_strings), approaches_count))
    for i in range(len(all_eval_metric_values)):
        inx1 = int(i%len(config.evaluation_metrics_strings))
        inx2 = int(i/len(config.evaluation_metrics_strings))
        array[inx1][inx2] = round(all_eval_metric_values[i],2)

    csv_scores.update({"Approaches": approachesList}) 

    for i in range(len(config.evaluation_metrics_strings)):
        csv_scores.update({f"{config.evaluation_metrics_strings[i]}": list(array[i])}) 
        
    df = pd.DataFrame(csv_scores)
    csv_name = f'{dataset_name}_{vectorizer}.csv'
    
 
    # Write DataFrame to CSV File with Default params.
    df.to_csv(os.path.join(config.csv_dir, csv_name), index = False) #, a_rep = 'null'


def clust_algo_to_csv(clustering_algorithms_string, parameters, arguments):
    if (type(arguments[0]) is int): 
        return reduce(
            lambda x,y: f"{x}|{y}", [f"{clustering_algorithms_string}"] + [f"{a}:{b}" for a, b in zip(parameters[1:], map(str, arguments[1:]))] 
        )
    return reduce(
        lambda x,y: f"{x}|{y}", [f"{clustering_algorithms_string}"] + [f"{a}:{b}" for a, b in zip(parameters, map(str, arguments))]   
    )

def plot_histogram(x_list, dataset_string):

    value_counts = Counter(x_list)

    # Separate the values and their counts
    values = list(value_counts.keys())
    counts = list(value_counts.values())

    # Combine the two lists using zip
    #combined_lists = list(zip(values, counts))
    #sorted_combined_lists = sorted(list(zip(values, counts)), key=lambda x: x[0])
    values_sorted, counts_sorted = zip(*sorted(list(zip(values, counts)), key = lambda x: x[0]))

    plt.figure(figsize=(9, 4))
    plt.bar(values_sorted, counts_sorted, width = 0.5)
  
    plt.xticks(values_sorted, values_sorted)
    plt.ylabel('Count')
    plt.xlabel('True Labels')
    plt.title("".join(['True Labels Distribution for <<', dataset_string.upper(), ">>"]))
    plt.grid(True)
    plt.xticks(rotation=45)
    for x, y in zip(values, counts):
        plt.text(x, y, str(y), ha='center', va='bottom')
    plt.show()
    

def create_serialized_vectors_dirs():
    """
    Create folder for each dataset for each vectorize approach
    to store pickle files for each document
    """
    path = "precomputed_vectors\\"
    if not os.path.exists(path):
        os.makedirs(path)

    for datasets_folder in config.datasets_strings:
        if not os.path.exists("".join([path,datasets_folder,"\\"])):
            os.makedirs("".join([path,datasets_folder,"\\"])) 
    
    for datasets_folder in config.datasets_strings:
        for vectorizer_folder in config.vectorizers_strings:
            if not os.path.exists("".join([path,datasets_folder,"\\",vectorizer_folder])):
                os.makedirs("".join([path,datasets_folder,"\\",vectorizer_folder]))


def check_folder_size(dataset_string, vectorizer_string):
    path = "precomputed_vectors\\"
    path = "".join([path,dataset_string,"\\",vectorizer_string,"\\"])
    return os.path.getsize(path)
        

def store_serialized_vector(dataset_string, vectorizer_string, vectors, labels_true):
    file_path = f"precomputed_vectors\\{dataset_string}\\{vectorizer_string}\\labels_true.pkl"
    dbfile = open(file_path, "ab")
    pickle.dump(labels_true, dbfile)                     
    dbfile.close()

    file_path = f"precomputed_vectors\\{dataset_string}\\{vectorizer_string}\\shape.pkl"
    dbfile = open(file_path, "ab")
    pickle.dump(vectors.shape, dbfile)                     
    dbfile.close()

    for indx, vector in enumerate(vectors):
        file_path = f"precomputed_vectors\\{dataset_string}\\{vectorizer_string}\\{indx}.pkl"
        dbfile = open(file_path, "ab")
        pickle.dump(vector, dbfile)                     
        dbfile.close()


def load_deselialized_vector(dataset_string, vectorizer_string):
    file_path = f"precomputed_vectors\\{dataset_string}\\{vectorizer_string}\\shape.pkl"
    dbfile = open(file_path, 'rb')     
    shape = pickle.load(dbfile)
    dbfile.close()

    file_path = f"precomputed_vectors\\{dataset_string}\\{vectorizer_string}\\labels_true.pkl"
    dbfile = open(file_path, 'rb')     
    labels_true = pickle.load(dbfile)
    dbfile.close()

    arr = np.array([])
    for indx in range(shape[0]):
        file_path = f"precomputed_vectors\\{dataset_string}\\{vectorizer_string}\\{indx}.pkl"
        dbfile = open(file_path, "rb")
        vector = pickle.load(dbfile)     
        arr = np.append(arr, vector)            
        dbfile.close()

    arr = arr.reshape(shape)
    return arr, labels_true


def load_dataset_arff(dataset_string):

    token_freq_vectors, labels_true, vocabulary = ([], [], [])
    path_to_directory = config.local_datasets_path + dataset_string + "/"
    file_arff = dataset_string + ".arff"

    with open(path_to_directory + file_arff , "r") as inFile:
        dataset_arff = inFile.readlines()

        data_start = False
        for line in dataset_arff:
            if not data_start:
                if "@attribute" in line.lower():
                    vocab_line = line.split()
                    vocabulary.append(vocab_line[vocab_line.index("@attribute") +1 ])
                elif "@data" in line.lower():
                    print(line)
                    data_start = True
            else:
                line_indc = line.split(",")
                labels_true.append(line_indc[-1].strip('" \n'))
                del line_indc[-1]
                token_freq_vectors.append([int(x) for x in line_indc])
            
    n_clusters = len(set(labels_true))
    print("ok")
    return [token_freq_vectors, labels_true, n_clusters]


def load_dataset_mat(dataset_string):

    path_to_directory = config.local_datasets_path + dataset_string + "/"
    file_mat = dataset_string + ".mat"
    # file_mat = "CSTR_coclustFormat" + ".mat"
    matlab_dict = loadmat(path_to_directory + file_mat)
    doc_vectors = matlab_dict['fea'].tolist()
    labels_true = [label[0] for label in matlab_dict['gnd'].tolist()]

    print("--")

    return np.array(doc_vectors, dtype = object), labels_true
