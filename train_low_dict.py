from time import time
import numpy as np
import extract_data as ext
import sparse_code as sparse
from sklearn.preprocessing import Imputer
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import sparse_encode
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
dict_file = 'dictionary.pkl'

train_pixels = np.load('C:/Users/Timothy/HSI_Dictionaries/Data/Intermediate Data/' + 'low_data.npy')
print(train_pixels.shape)
train_pixels = train_pixels[0:1000,0:train_pixels.shape[1]]
train_pixels = normalize(train_pixels)
print(train_pixels.shape)
n_components = 250
alpha = 10
max_iterations = 1000
fit_algo = 'cd'
tol = 1e-6
"""
print('Finding dictionary')
t0 = time()

dico = DictionaryLearning(n_components=n_components, alpha=alpha, max_iter=max_iterations, tol=tol, transform_algorithm='omp' , fit_algorithm=fit_algo)

dictionay = dico.fit(train_pixels).components_
dt = time() - t0

joblib.dump(dico, dict_file)
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
