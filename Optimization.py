import numpy as np
from extract_data import get_spectral_library, normalize, convert_library, create_sample
from create_hsi import create_hsi
from sklearn.preprocessing import Imputer
from sklearn.decomposition import sparse_encode
K = 15
h = .025
mu = .5
lamb = .005
gamma = .005

library = get_spectral_library('AVIRIS2014','Minerals')
samples, names = create_sample(library)
samples = normalize(samples)

image = create_hsi(samples)
print(image.shape)

data = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
print(data.shape)

dictionary =  convert_library(library)
print(dictionary.shape)
imputer_data = Imputer()
imputer_dict = Imputer()
imputer_data.fit(data)
imputer_dict.fit(dictionary)
data = imputer_data.transform(data)
dictionary = imputer_dict.transform(dictionary)

#X = sparse_encode(data, )

V1 = np.ones((data.shape))
V2 = np.ones((data.shape))
V3 = np.ones((data.shape))
V4 = np.ones((data.shape))

V = [V1, V2, V3, V4]

D1 = np.ones((data.shape))
D2 = np.ones((data.shape))
D3 = np.ones((data.shape))
D4 = np.ones((data.shape))

D = [D1, D2, D3, D4]

