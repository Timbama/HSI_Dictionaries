from Optimization import morph_opt, reg_opt
import numpy as np
from extract_data import initialize_file
from skimage.morphology import square
from create_hsi import create_hsi
import matplotlib.pyplot as plt
mu = .004
lamb = .1
gamma = .006
n_iter = 1000
width = 5
strel = square(width)
M = initialize_file('USGS_pruned_10_deg.mat','B')
data = initialize_file('SNR 30 data.mat','Y')
actual = initialize_file('SNR 60 data.mat','Y')
actual = np.transpose(actual)
data = np.transpose(data)
Y = data
M = np.transpose(M)
print(np.amax(Y))
print(np.amin(Y))
sample = data[0:5,:]
#image = create_hsi(sample)
#print(image.shape)
#Y = np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2]))
print("Dictionary shape:", M.shape)
X_reg = reg_opt(M,Y,lamb,mu) 
X = morph_opt(M, Y, lamb, gamma, mu, strel, n_iter=4000)
M = np.transpose(M)
recon = np.dot(M,X)
recon_reg = np.dot(M,X_reg)
recon = np.transpose(recon)
recon_reg = np.transpose(recon_reg)
error = np.linalg.norm((actual-recon),'fro')
diff =np.abs(recon-actual)
avg_error = np.mean(diff)
max_error = np.amax(diff)
std_error = np.std(diff)
error_reg = np.linalg.norm((actual-recon_reg),'fro')
diff_reg =np.abs(recon_reg-actual)
avg_error_reg = np.mean(diff_reg)
max_error_reg = np.amax(diff_reg)
std_error_reg = np.std(diff_reg)
#recon = np.reshape(recon, (75,75,224))
print("Norm of the difference Morph", error)
print("average error Morph", avg_error)
print("Standad deviation of error Morph", std_error)
print("max error Morph", max_error)
print("Norm of the difference regular", error_reg)
print("average error regular", avg_error_reg)
print("Standad deviation of error regular", std_error_reg)
print("max error regular", max_error_reg)
plt.subplot(311); plt.imshow(actual); plt.title('Origional')
plt.subplot(312); plt.imshow(recon); plt.title('reconstruction')
plt.subplot(313); plt.imshow(recon_reg); plt.title('regular')
plt.show()