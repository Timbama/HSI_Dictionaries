from time import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning
import tifffile as tif
import random

data = tif.imread('19920612_AVIRIS_IndianPine_Site3.tif')
print(data.shape)
wavelengths = np.zeros(200)
dataFile = open('wavelengths.csv', 'r')
temp = [list(map(float,rec)) for rec in csv.reader(dataFile, delimiter=',')]
wavelengths = temp[0]
print(len(wavelengths))
dim = (145,145)
sample = np.empty((0,220))
#sample = np.zeros(200)
print('Selecting a training set... ')
t0 = time()
i=0
for i in range(500):
    randX = random.randint(0,dim[0]-1)
    randY = random.randint(0,dim[1]-1)
    sample = np.append(sample, [data[:,randX,randY]], axis=0)
    i+=1
dt = time() - t0
print('done in %.2fs.' % dt)
print(sample.shape)

print('Learning Dictionary')
t0 = time()
dico = DictionaryLearning(n_components=500, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm='cd', transform_algorithm='omp', n_jobs=1, verbose=True)
V = dico.fit(sample).components_
dt = time() - t0
print('done in %.2fs.' % dt)
print(V.shape)