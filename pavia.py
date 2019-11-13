#%%
import os
import importlib
import Optimization
import hyperspy.api as hs
import matplotlib
#matplotlib.rcParams["backend"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import square, disk
from sklearn.preprocessing import Imputer
from spectral import imshow, view_cube

from create_hsi import create_hsi
from evaluate import calculate_rsme, compare_abundance
from extract_data import (convert_library, get_spectral_library,
                          initialize_file, normalize, prune_library,
                          remove_bands)

from segmentation import create_SAD_mat
from USG_data_paths import dataPath, dataPath_HSI
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import normalize

import hyperspy.api as hs
hs.preferences.GUIs.enable_ipywidgets_gui = True
hs.preferences.GUIs.enable_traitsui_gui = False
hs.preferences.save()


#%%
# Load the Ground truth images, endmembers, and abundances and reshape them correctly
image_gt = initialize_file(os.path.join(dataPath_HSI,'Pavia','PaviaU.mat'), key='paviaU')
test_size = 75
image_gt = image_gt[:test_size,:test_size,:]
print("Collection Size:", image_gt.shape, "(width x length x number of bands)")


# Reshape the image_gt to be a set of matrcies
image_flat = np.reshape(image_gt, (image_gt.shape[0]*image_gt.shape[1],image_gt.shape[2]))
image_flat = np.swapaxes(image_flat,0,1)
image_flat = normalize(image_flat)
#image_flat = np.divide(image_flat,np.max(image_flat))
#image_gt = np.reshape(image_flat,image_gt.shape)
print("Flat Images Size:", image_flat.shape, "(number of samples x number of bands)")
#test_size = 200
#image_flat = image_flat[:,:1000]

#%%
importlib.reload(Optimization)

mu = .2
lamb = .3
gamma = .6
n_iter = 100
operation = "closing"

ind = np.arange( image_flat.shape[ 1 ] )
np.random.shuffle( ind )
Dict = image_flat[:,ind[:25]]
Dict = Dict.astype(float)
min_rsme = 1000
for i in range(1,2):
    width = i
    strel = square(width)
    X_reg,M = Optimization.morph_opt(Dict, image_flat, lamb, gamma, mu, strel, operation, n_iter, verbose=False)
    print("Abundance Fraction Size:", X_reg.shape, "(number of endmembers x number of samples)")
    rsme = calculate_rsme(np.dot(M, X_reg), image_flat)
    print("The RSME is ", rsme)
    if (rsme  < min_rsme):
        min_rsme = rsme
        rsme_index = i
print("Best index is ", rsme_index)
#%%
visual = np.dot(M, X_reg)
visual = np.swapaxes(visual,0,1)
visual = np.reshape(visual, (image_gt.shape[0], image_gt.shape[1], image_gt.shape[2]))
s = hs.signals.Signal1D(visual)
s.plot()
#image_gt = np.swapaxes(np.swapaxes(image_gt,0,2),0,1)
image_display = hs.signals.Signal1D(image_gt)
image_display.plot()
#input()
#%%
class_gt  = initialize_file(os.path.join(dataPath_HSI,'Pavia','PaviaU_gt.mat'), key='paviaU_gt')
class_gt  = class_gt[:test_size,:test_size]
class_gt_flat = np.reshape(class_gt, class_gt.shape[0]*class_gt.shape[1])
X = np.swapaxes(X_reg,0,1)
train_samples = 25
ind = np.arange( image_flat.shape[ 1 ] )
np.random.shuffle( ind )
X_train = X[ind[:train_samples],:]
X_test = X[ind[train_samples:],:]
y_train = class_gt_flat[ind[:train_samples]]
y_test = class_gt_flat[ind[train_samples:]]
print(y_test.shape)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
acc  = rfc.score(X_test,y_test)
print(acc)
# %%
