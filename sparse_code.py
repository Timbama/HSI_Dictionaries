import numpy as np

from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import SparseCoder

def initialize_dict(n_components, n_features):
    n_nonzero_coefs = 20
    y, X, w = make_sparse_coded_signal(n_samples = 1, n_components=n_components, n_features=n_features, n_nonzero_coefs=n_nonzero_coefs)
    return X
    
def omp_sparse(dictionary, train_data):

    dictionary = dictionary.transpose()
    
    w = SparseCoder(dictionary, transform_algorithm='omp')
    t = w.transform(train_data)
    return t
def find_nonzero(sparse_codes):
    index = np.where(sparse_codes != 0)
    return index