import numpy as np
from USG_data_paths import dataPath_HSI
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import sparse_encode
from sklearn.preprocessing import normalize
from time import time
n_samples = 1000
n_components = 200
tol = 1e-3
max_iter = 1000
print('Loading, normalizing and selecting training data')
t0 = time()
#load data
data = np.load(dataPath_HSI  + 'high_data.npy')
#normalize data set
data_norm = (data-np.amin(data))/(np.amax(data)-np.amin(data))
#select data to train dictionary
random_index = np.random.choice(data_norm.shape[0],size=n_samples,replace=False)
train_data  = data_norm[random_index,:]
dt = time() - t0
print('done in %.2fs.' % dt)
dictionary = data_norm[0:n_components,:]
code = sparse_encode(train_data, dictionary, algorithm='omp',max_iter=max_iter, n_nonzero_coefs=100)
print(code.shape)
residual = train_data - np.matmul(code, dictionary)
print(residual.shape)
print(np.mean(residual))
for i in range(2):
    I = np.nonzero(code)[0]
    print(I)
    residual_i = residual[I]
    print(residual_i.shape)

