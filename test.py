import numpy as np
from USG_data_paths import dataPath_HSI
import matplotlib.pyplot as plt
from extract_data import get_spectral_library, create_sample, normalize, convert_library, prune_library
from create_hsi import create_hsi, abundance_map
from sklearn.decomposition import sparse_encode
from sklearn.preprocessing import Imputer
library = get_spectral_library('AVIRIS2014','Minerals')
samples, names = create_sample(library)
print(np.nanmax(samples))
print(np.nanmin(samples))
samples = normalize(samples)
print(samples.shape)
print(np.nanmax(samples))
print(np.nanmin(samples))
print(len(library))

image = create_hsi(samples)
image_test = create_hsi(samples)
print(image.shape)
abundance = abundance_map((.5, .33333, .25, .2), 1, (75,75))
data = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
print(data.shape)
dictionary, keys =  convert_library(library)
print(dictionary.shape)
imputer_data = Imputer()
imputer_dict = Imputer()
imputer_data.fit(data)
imputer_dict.fit(dictionary)
data = imputer_data.transform(data)
dictionary = imputer_dict.transform(dictionary)



sparse = sparse_encode(data, dictionary, algorithm='lasso_cd', max_iter=1000, n_nonzero_coefs=20, alpha=2)

print(sparse.shape)

#output = np.reshape(sparse, (75,75,224))
numbers = []
for n in names:
    iter = 0
    for keys in sorted(library.keys()):
        iter += 1
        if n == keys:
            numbers.append(iter)
print(numbers)
used = np.zeros((data.shape[0],0))
for num in numbers:
    temp = np.reshape(sparse[:,num], (sparse.shape[0],1))
    used = np.hstack((used, temp))
    print(np.sum(temp))
print(used.shape)

image = np.reshape(used,(75,75,5))
image = np.reshape(used[:,2], (75,75))
print(used)
#for i in range(5): 
    #print(np.sum(image[:,:,i]))
plt.imshow(image[:,:])
plt.imshow(image_test[:,:,20])
plt.imshow(abundance)
plt.show()