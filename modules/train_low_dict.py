from time import time
import numpy as np
import extract_data as ext
import sparse_code as sparse
from sklearn.preprocessing import Imputer
from sklearn.decomposition import DictionaryLearning
from sklearn.externals import joblib
dict_file = 'dictionary.pkl'

train_pixels = np.load('G:/timba/Documents/Hyperspectral project/Data/Intermediate Data/' + 'low_data.npy')
n_components = 150
alpha = 10
max_iterations = 1000
fit_algo = 'cd'
tol = 1e-6
print('Finding dictionary')
t0 = time()

dico = DictionaryLearning(n_components=n_components, alpha=alpha, max_iter=max_iterations, tol=tol, transform_algorithm='omp' , fit_algorithm=fit_algo)

dictionay = dico.fit(train_pixels).components_
dt = time() - t0

joblib.dump(dico, dict_file)