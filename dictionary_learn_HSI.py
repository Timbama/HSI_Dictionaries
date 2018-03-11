from time import time
import numpy as np
import extract_data as ext
#from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning
from sklearn.externals import joblib
#File name for dictionary model
dict_file = 'dictionary.pkl'
#Data path in file
dataPath = 'G:/timba/Documents/Hyperspectral project/Data/'
filename = dataPath + 'PaviaU.mat'
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
print(np.amax(data))
print(np.amin(data))
print(np.mean(data))
print(np.std(data))
#Select a training set
print('Selecting a training set... ')

train_pixels = ext.extract_pixel_random(numOfTrain, data)
dt = time() - t0
print('done in %.2fs.' % dt)
print(train_pixels)


print('Learning Dictionary')
t0 = time()
dico = DictionaryLearning(n_components=300, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm='cd', transform_algorithm='omp', n_jobs=1, verbose=True)
dico.fit(train_pixels)
dt = time() - t0
print('done in %.2fs.' % dt)

joblib.dump(dico, dict_file)

print('Saved Dictionary to file ' + dict_file)