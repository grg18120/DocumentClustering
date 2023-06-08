from sklearn import metrics 
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np

"""
External evaluation, clustering is compared to an existing "ground truth" classification
"""

def rand_index(labels_true, labels_pred):
	"""
	Rand Index [0, 1] 
	"""
	return metrics.rand_score(labels_true, labels_pred)
		

def adjusted_rand_index(labels_true, labels_pred):
	"""
	Adjusted Rand Index [-1, 1] 
	"""
	return metrics.adjusted_rand_score(labels_true, labels_pred)


def fowlkes_mallows_index(labels_true, labels_pred):
	"""
	Fowlkes-Mallows metric
	Values close to zero indicate two label assignments that are largely independent, 
	while values close to one indicate significant agreement. 
	Further, values of exactly 0 indicate purely independent label assignments and a FMI of exactly 1 indicates that 
	the two label assignments are equal 
	"""
	return metrics.fowlkes_mallows_score(labels_true, labels_pred)


def v_measure_index(labels_true, labels_pred):
	"""
	Homogeneity(h), completeness(c) and V-measure metric
	V-measure = [(1+b)*h*c] / [(b*h + c)]   b = 1.0(default)
	0.0 is as bad as it can be, 1.0 is a perfect score.
	h = metrics.homogeneity_score(labels_true, labels_pred)
	c = metrics.completeness_score(labels_true, labels_pred)
	v = ((1+b)*h*c) / ((b*h + c))   b = 1.0
	"""
	return metrics.v_measure_score(labels_true, labels_pred, beta = 1.0) 


def homogenity(labels_true, labels_pred):
	"""
	Cluster contain only data points which are members of a single class
	All members of a given class are assigned to the same cluster
	How much the samples in a cluster are similar.
	"""
	return metrics.homogeneity_score(labels_true, labels_pred)


def completeness(labels_true, labels_pred):
	"""
	Class points are elements of the same cluster
	How much similar samples are put together by the clustering algorithm.
	"""
	return metrics.completeness_score(labels_true, labels_pred)


def mutual_information(labels_true, labels_pred):
	return metrics.mutual_info_score(labels_true, labels_pred)


def adjusted_mutual_information(labels_true, labels_pred):
	return metrics.adjusted_mutual_info_score(labels_true, labels_pred)


def normalized_mutual_information(labels_true, labels_pred):
	return metrics.normalized_mutual_info_score(labels_true, labels_pred)


def pairwise_distances(cl_a, cl_b, metric):
	return metrics.pairwise_distances(cl_a, cl_b, metric = metric)


def f1_score(labels_true, labels_pred):
	return metrics.f1_score(labels_true, labels_pred, average = 'macro')


def f1_score_relabel(labels_true, labels_pred):

	def make_cm_cost(cm):
		s = np.max(cm)
		return (- cm + s)
	
	# Permutation maximizing the sum of the diagonal elements - apply Hungarian algorithm
	cm = metrics.confusion_matrix(labels_true, labels_pred)
	indexes = linear_assignment(make_cm_cost(cm))
	change_from = indexes[0].tolist()
	change_to = indexes[1].tolist()
	can_update = [1] * len(labels_true)
	
	labels_pred_relabel = labels_pred[:]
	for x_index,x in enumerate(change_from):
		for y_index,y in enumerate(labels_pred):
			if y == x and can_update[y_index] == 1:
				can_update[y_index] = -1
				labels_pred_relabel[y_index] = change_to[x_index] 

	return metrics.f1_score(labels_true, labels_pred_relabel, average = 'macro')


def jaccard_score(labels_true, labels_pred):
	return metrics.jaccard_score(labels_true, labels_pred, average = 'macro')