from time import time
import numpy as np
import extract_data as ext
import sparse_code as sparse
from sklearn.preprocessing import Imputer
from sklearn.decomposition import DictionaryLearning, sparse_encode, dict_learning_online
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_sparse_coded_signal
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from USGS_data_path_Laptop import dataPath_HSI
dict_file = 'dictionary.pkl'

train_pixels = np.load(dataPath_HSI + 'low_data.npy')
print(train_pixels.shape)
train_pixels = train_pixels[0:1000,0:train_pixels.shape[1]]
train_pixels = normalize(train_pixels)
print(train_pixels.shape)
n_components = 250
alpha = 10
max_iterations = 1000
fit_algo = 'cd'
tol = 1e-6


n_features = train_pixels.shape[1]
n_samples = train_pixels.shape[0]
n_nonzero_coefs = 10

print("Initilizing data set")
t0 = time()
code, low_dictionary = dict_learning_online(train_pixels, n_components=n_components, alpha=1, n_iter=max_iterations, method='cd', verbose=False)
dt = time() - t0
print('done in %.2fs.' % dt)






"""
#log file

n_features = train_pixels.shape[1]
n_samples = train_pixels.shape[0]
n_nonzero_coefs = 10

D = train_pixels[0:n_components,0:train_pixels.shape[1]]
D_trans = D.transpose()
print(D.shape)
train_pixels_trans = train_pixels.transpose()
omp = OrthogonalMatchingPursuit(tol=tol, normalize=True, fit_intercept=True)
omp.fit(D_trans, train_pixels_trans)

code = omp.coef_
print(code.shape)
print(D.shape)
print(train_pixels.shape)
for x in range(D.shape[0]):
    coef_atom = code[:,x]
    index = np.zeros(0, dtype=np.int64)
    for y in range(n_samples):
        if coef_atom[y] != 0:
            index = np.append(index, y)
    print(index.shape)
    #extract data for each atom in D
    update = train_pixels[index,:]

print(code.shape)
"""