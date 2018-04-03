import numpy as np
from skimage.morphology import disk, erosion, dilation
from sklearn.preprocessing import normalize
from sklearn.decomposition import sparse_encode
from extract_data import initialize_file
from USG_data_paths import dataPath_HSI

data = initialize_file(dataPath_HSI + 'PaviaU.mat')
data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]), order='C')
print(data.shape)
data = data[0:1000, :]
print(data.shape)
circle = disk(2)
print(circle)
data = normalize(data)

dictionary = np.load('dictionary.npy')

dictionary = normalize(dictionary)

code = sparse_encode(data, dictionary, algorithm='omp',verbose=100)

