
import matplotlib
from Optimization import morph_opt, reg_opt
import numpy as np
from extract_data import initialize_file, normalize, get_spectral_library, convert_library,remove_bands, prune_library
from skimage.morphology import square
from create_hsi import create_hsi
import matplotlib.pyplot as plt
from spectral import view_cube
from sklearn.preprocessing import Imputer
#Setup Constants

mu = .1
lamb = .1
gamma = .1
n_iter = 1000
width = 2
strel = square(width)
#Create dictionary, data, and comparitive data
library = get_spectral_library('AVIRIS1995', 'Minerals')
library = prune_library(library,3)
M, names = convert_library(library)
dict_imp = Imputer()
dict_imp.fit(M)
M = dict_imp.transform(M)
M = normalize(M)
M = M.transpose()
print("Dictionary shape:", M.shape)
data = M #shape: bands x samples
print("Data shape:", data.shape)
actual = M #shape: bands x samples
sample = data[:,0:5]
sample = sample.transpose()
print(sample.shape)
image = create_hsi(sample)
print(image.shape)
image = np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2]))
print(image.shape)

#view_cube(image, bands=[29, 19, 9])
X_reg = reg_opt(M, image,lamb,mu) 
X = morph_opt(M, image, lamb, gamma, mu, strel, n_iter=2000)
recon = np.dot(M,X)
recon_reg = np.dot(M,X_reg)

#Calculate the error (morph)
error = np.linalg.norm((actual-recon),'fro')
diff =np.abs(recon-actual)
avg_error = np.mean(diff)
max_error = np.amax(diff)
std_error = np.std(diff)
#Calculate the error (reg)
error_reg = np.linalg.norm((actual-recon_reg),'fro')
diff_reg =np.abs(recon_reg-actual)
avg_error_reg = np.mean(diff_reg)
max_error_reg = np.amax(diff_reg)
std_error_reg = np.std(diff_reg)
recon = np.reshape(recon, (75,75,224))


