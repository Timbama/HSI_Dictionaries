#%%
import os
import importlib
import Optimization
import hyperspy.api as hs
import matplotlib
matplotlib.rcParams["backend"] = "Agg"
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

import hyperspy.api as hs
hs.preferences.GUIs.enable_ipywidgets_gui = True
hs.preferences.GUIs.enable_traitsui_gui = False
hs.preferences.save()


#%%
# Load the Ground truth images, endmembers, and abundances and reshape them correctly
images_gt = initialize_file(os.path.join(dataPath_HSI,'Legendre','5e128x128LegendreDefault.mat'), key='syntheticImage')
print("Collection Size:", images_gt.shape, "(width x length x number of bands)")
endmembers_gt = initialize_file(os.path.join(dataPath_HSI,'Legendre','5e128x128LegendreDefault.mat'), key='endmembersGT')
endmembers_gt = np.swapaxes(endmembers_gt, 0,1)
print("Endmembers Size:", endmembers_gt.shape, "(number of endmembers x number of bands)")
abundances_gt = initialize_file(os.path.join(dataPath_HSI,'Legendre','5e128x128LegendreDefault.mat'), key='abundanciesGT')
print("Abundances Size:", abundances_gt.shape, "(width x length x number of endmembers)")

# Reshape the images_gt to be a set of matrcies
images_flat = np.reshape(images_gt, (images_gt.shape[0]*images_gt.shape[1],images_gt.shape[2]))
images_flat = np.swapaxes(images_flat,0,1)
print("Flat Images Size:", images_flat.shape, "(number of samples x number of bands)")

# Reshape Data to be a set of abundance matricies
abundance_flat = np.reshape(abundances_gt, (abundances_gt.shape[0]*abundances_gt.shape[1],abundances_gt.shape[2]))
abundance_flat = np.swapaxes(abundance_flat,0,1)
print("Abundances Reshape Size:", abundance_flat.shape, "(number of samples x number of endmembers)")


print(np.max(endmembers_gt))
#%%
importlib.reload(Optimization)
mu = .1
lamb = .1
gamma = .1
n_iter = 100
width = 5
strel = square(width)
for i in range(1,2):
    X_reg = Optimization.morph_opt(endmembers_gt, images_flat, lamb, gamma, i*mu, strel, n_iter)
    print("Abundance Fraction Size:", X_reg.shape, "(number of endmembers x number of samples)")
    avg = compare_abundance(abundance_flat, X_reg)
    print("The RSME is ", avg)
#%%

visual = np.dot(endmembers_gt, X_reg)
visual = np.reshape(visual, (images_gt.shape[2], images_gt.shape[1], images_gt.shape[0]))
s = hs.signals.Signal2D(visual)
s.plot()

image_display = hs.signals.Signal2D(np.swapaxes(np.swapaxes(images_gt,0,2),1,2))
image_display.plot()
#input()
train_samples = 100
X_train = X_reg[:train_samples]
X_test = X_reg[train_samples:]



rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

# %%
