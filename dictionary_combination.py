from time import time
import numpy as np
import extract_data as ext
import sparse_code as sparse
from sklearn.preprocessing import Imputer
#File name for dictionary model
dict_file = 'dictionary.pkl'
#Data path in file
dataPath = 'G:/timba/Documents/Hyperspectral project/Data/'
#dataPath = 'G:/timba/Documents/OneDrive for Business/OneDrive - Kennesaw State University/data/ASCIIdata'
filename = dataPath + 'PaviaU.mat'
n_components = 150
#the number of trian samples
numOfTrain = 250
testX = 10
testY = 10
#import the data and wavelengths
print("Initilizing data set")
t0 = time()
data = ext.initialize_file(filename)
dt = time() - t0
print('done in %.2fs.' % dt)
print("Initialized data set with shape" , data.shape)

#Select a training set
print('Selecting a training set... ')
train_pixels = ext.extract_pixel_random(numOfTrain, data)
dt = time() - t0
print('done in %.2fs.' % dt)
print(train_pixels.shape)

#select a set of test spectra from the USGS database
print('selecting training set for USGS data')
t0 = time()
d = ext.get_spectra('Vegetation', 'ASD')

imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0,)
imp.fit(d)
d = imp.transform(d)
print(d.shape)
dt = time() - t0
print('done in %.2fs.' % dt)

print('Find Sparse Representation for both scales')
t0 = time()
dict1 = sparse.initialize_dict(n_components, d.shape[1])
dict2 = sparse.initialize_dict(n_components, train_pixels.shape[1])


code_mat_1 = np.array([]).reshape(0,n_components)
code_mat_2 = np.array([]).reshape(0, n_components)

for x in range(d.shape[0]):
    print('sparsifying sample ', x+1, ' for dictionary 1')
    code = sparse.omp_sparse(dict1, d[x,:].reshape(1, d.shape[1]))
    code_mat_1 = np.vstack([code_mat_1, code])
for x in range(train_pixels.shape[0]):
    print('sparsifying sample ', x+1, ' for dictionary 2')
    code = sparse.omp_sparse(dict2, train_pixels[x,:].reshape(1, train_pixels.shape[1]))
    code_mat_2 = np.vstack([code_mat_2, code])
dt = time() - t0
print('done in %.2fs.' % dt)
print(code_mat_1.shape, code_mat_2.shape)

print('Find an index set for the non zero compenets of a')

index1 = sparse.find_nonzero(code_mat_1)
print(index1)