import numpy as np
from pywt import threshold

dictionary = np.load('dictionary.npy')
print(dictionary.shape)