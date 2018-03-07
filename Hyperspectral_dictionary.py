from time import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import extract_data as ext
#from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import DictionaryLearning


dataPath = 'G:/timba/Documents/Hyperspectral project/Data/'
refl_filename = dataPath + '19920612_AVIRIS_IndianPine_Site3.tif'
wave_filename = dataPath + 'wavelengths.csv'
#the number of trian samples
numOfTrain = 5000
testX = 10
testY = 10
#import the data and wavelengths
data, wavelengths = ext.initialize_tiff(refl_filename, wave_filename)
print(len(wavelengths))
dim = (145,145)
print(ext.extract_pixel(1,1,data))
#Select a training set
print('Selecting a training set... ')
t0 = time()
train_pixels = ext.extract_pixel_random(numOfTrain, dim, data)
train_pixels -= np.mean(train_pixels, axis=0)
train_pixels /= np.std(train_pixels, axis=0)
print(train_pixels)
dt = time() - t0
print('done in %.2fs.' % dt)
print(train_pixels.shape)

print('Learning Dictionary')
t0 = time()
dico = DictionaryLearning(n_components=300, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm='cd', transform_algorithm='omp', n_jobs=1, verbose=True)
V = dico.fit(train_pixels).components_
dt = time() - t0
print(V.shape)
print('done in %.2fs.' % dt)

print("finding test pixel")
t0 = time()
noise = np.random.normal(0,1,220)
test_pixel = ext.extract_pixel(1, 1, data)
#test_pixel = test_pixel - np.mean(test_pixel, axis=0)
print(test_pixel)
corrupt_test_pixel = test_pixel + noise
dt = time() - t0
print(np.mean(test_pixel))
print('done in %.2fs.' % dt)

print("find sparse representation of a pixel")
t0 = time()
test_sparse = dico.transform(corrupt_test_pixel)
dt = time() - t0
print(test_sparse)
print("the sparsity is: ", np.count_nonzero(test_sparse))
print('done in %.2fs.' % dt)

print("recover the value of the test pixel and compare it to the actual")
t0 = time()
recovered_pixel = np.dot(test_sparse, V)
print(recovered_pixel)
actual_pixel = test_pixel
#Calculate the Error from the reconstruction
error = actual_pixel - recovered_pixel
avg_error = np.mean(error)
dt = time() - t0
print('done in %.2fs.' % dt)
print("average error", avg_error)

#Setup the pixels as data frames and plot the spectrums for comparison
pixel_df = pd.DataFrame(index=wavelengths)
pixel_df['Actual'] = actual_pixel[0]
pixel_df['Reconstruction'] = recovered_pixel[0]
plt.figure(); pixel_df.plot(); plt.legend(loc='best')
plt.title('Spectral Signature for TALL pixel (5000,500)')
ax = plt.gca()
ax.set_xlim([np.min(wavelengths),np.max(wavelengths)])
ax.set_ylim([np.min(pixel_df['Actual']), np.max(pixel_df['Actual'])])
ax.set_xlabel("Wavelength, nm"); ax.set_ylabel("Reflectance")
ax.grid('on')
plt.show()
