
import matplotlib
from Optimization import morph_opt, reg_opt
import numpy as np
from extract_data import initialize_file
from skimage.morphology import square
from create_hsi import create_hsi
import matplotlib.pyplot as plt
from spectral import view_cube
#Setup Constants

mu = .1
lamb = .1
gamma = .1
n_iter = 1000
width = 2
strel = square(width)
#Create dictionary, data, and comparitive data
dictionary = initialize_file('USGS_pruned_10_deg.mat','B') #shape: bands x atoms
print("Dictionary shape:", dictionary.shape)
data = initialize_file('SNR 10 data.mat','Y') #shape: bands x samples
print("Data shape:", data.shape)
actual = initialize_file('SNR 60 data.mat','Y') #shape: bands x samples
sample = data[:,0:5]
print(sample.shape)
image = create_hsi(sample)
print(image.shape)
image = np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2]))
print(image.shape)

#view_cube(image, bands=[29, 19, 9])
X_reg = reg_opt(dictionary, image,lamb,mu) 
X = morph_opt(dictionary, image, lamb, gamma, mu, strel, n_iter=2000)
recon = np.dot(dictionary,X)
recon_reg = np.dot(dictionary,X_reg)

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
print("Norm of the difference Morph", error)
print("average error Morph", avg_error)
print("Standard deviation of error Morph", std_error)
print("max error Morph", max_error)
print("Norm of the difference regular", error_reg)
print("average error regular", avg_error_reg)
print("Standard deviation of error regular", std_error_reg)
print("max error regular", max_error_reg)
plt.subplot(311); plt.imshow(actual); plt.title('With Noise')
plt.subplot(312); plt.imshow(recon); plt.title('reconstruction')
plt.subplot(313); plt.imshow(recon_reg); plt.title('regular')
plt.show()