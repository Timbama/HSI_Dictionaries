import numpy as np
from sklearn.decomposition import DictionaryLearning
data = np.load('PaviaU.npy')
n_components = 500
dictionary  = DictionaryLearning(n_components=n_components, max_iter=1000, fit_algorithm='cd',tol=1e-8, verbose=True)
dictionary.fit(data)
test = dictionary.components_
np.save('dictionary_all',test)
