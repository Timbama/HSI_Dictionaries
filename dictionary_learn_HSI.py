from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import extract_data as ext
#from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning
from sklearn.externals import joblib
import pickle
#File name for dictionary model
dict_file = 'dictionary.pkl'
#Data path in file
dataPath = 'G:/timba/Documents/Hyperspectral project/Data/'
#filename = dataPath + '19920612_AVIRIS_IndianPine_Site3.tif'
filename = dataPath + 'PaviaU.mat'
#the number of trian samples
numOfTrain = 5000
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

print('Learning Dictionary')
t0 = time()
dico = DictionaryLearning(n_components=300, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm='cd', transform_algorithm='omp', n_jobs=1, verbose=True)
dico.fit(train_pixels)
dt = time() - t0
print('done in %.2fs.' % dt)

joblib.dump(dico, dict_file)

print('Saved Dictionary to file ' + dict_file)