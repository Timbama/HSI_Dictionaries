import matplotlib
from Optimization import morph_opt, reg_opt
import numpy as np
from extract_data import initialize_file, normalize, get_spectral_library, convert_library,remove_bands, prune_library
from skimage.morphology import square
from create_hsi import create_hsi
import matplotlib.pyplot as plt
from spectral import view_cube, imshow
from sklearn.preprocessing import Imputer
from evaluate import compare_abundance, calculate_rsme
import hyperspy.api as hs
from USG_data_paths import dataPath_HSI
from segmentation import create_SAD_mat

mu = .5
lamb = .1
gamma = .1
n_iter = 200
width = 2
strel = square(width)

#collection = initialize_file('Legendre.mat', key='syntheticImageCollection')
#print("Collection Size:", collection.shape, "(number of bands x width x length x number of images)")
endmembers_gt = initialize_file('/mnt/c/Users/timba/HSI_Dictionaries/Data/hypespectral_images/hyperspectral.mat', key='endmembers')
print("Endmembers Size:", endmembers_gt.shape, "(number of bands x number of endmembers x number of images)")
abundances_gt = initialize_file('Data/hypespectral_images/hyperspectral.mat', key='abundancies')
print("Abundances Size:", abundances_gt.shape, "(number of endmembers x width x length)")
abundance_reshape = np.reshape(abundances_gt, (abundances_gt.shape[0], abundances_gt.shape[1]*abundances_gt.shape[2]))
print("Abundances Reshape Size:", abundance_reshape.shape, "(number of samples x number of endmembers)")
gt = np.zeros((endmembers_gt.shape[1], abundances_gt.shape[1]*abundances_gt.shape[2]))
gt[:5, :] = abundance_reshape
print("Abundance groundtruth reshape Size:", gt.shape, "(number of samples x number of endmembers)")
img = initialize_file('hyperspectral.mat', key='syntheticImage')
print("Image Size:", img.shape, "(number of endmembers x width x length)")
sim_map = create_SAD_mat(img)
lib = endmembers_gt
print("Library Size:", lib.shape, "(number of bands x number of endmembers )")
data = np.reshape(img, (img.shape[0], img.shape[1]*img.shape[2]))
print("Data Size:", data.shape, "(number of bands x number of samples)")

print("Similarity Map Size:", sim_map.shape)

#X_reg = morph_opt(lib, data, lamb, gamma, mu, strel, n_iter)
#print("Abundance Fraction Size:", X_reg.shape, "(number of endmembers x number of samples)")
#avg = compare_abundance(gt, X_reg)
#print("The RSME is ", avg)
#visual = np.dot(lib, X_reg)
#visual = np.reshape(visual, (img.shape[0], img.shape[1], img.shape[2]))
#s = hs.signals.Signal2D(visual)
#s.plot()
#input()