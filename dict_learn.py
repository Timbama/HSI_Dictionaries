import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import normalize
from extract_data import initialize_file
from USG_data_paths import dataPath_HSI

data = initialize_file(dataPath_HSI + 'PaviaU.mat')
data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]), order='C')
print(data.shape)
data = data[0:1000, :]
print(data.shape)

dict_obj = DictionaryLearning(n_components=100, tol=1e-3)
dict_obj.fit(data)

dictionary = dict_obj.components_

np.save('dictionary', dictionary)