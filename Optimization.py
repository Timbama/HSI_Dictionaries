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
n_iter = 1000
library = get_spectral_library('AVIRIS2014','Minerals')
samples, names = create_sample(library)
samples = normalize(samples)

image = create_hsi(samples)
print(image.shape)

Y = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
print(Y.shape)

dictionary =  convert_library(library)
print(dictionary.shape)
imputer_data = Imputer()
imputer_dict = Imputer()
imputer_data.fit(Y)
imputer_dict.fit(dictionary)
data = imputer_data.transform(Y)
M = imputer_dict.transform(dictionary)

X = sparse_encode(data, dictionary)

shape = X.shape()

v1 = M*X
v2 = X
v3 = X
v4 = X

d1 = np.zeros(shape)
d2 = np.zeros(shape)
d3 = np.zeros(shape)
d4 = np.zeros(shape)
i = 0
while i < n_iter:

    if i%10 == 0:
        v10 = v1
        v20 = v2
        v30 = v3
        v40 = v4
    v1 = (Y + mu*(M*X-d1))/(mu+1)



