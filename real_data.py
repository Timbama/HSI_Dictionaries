import matplotlib
matplotlib.use('WXAgg') 
import numpy as np
from Optimization import morph_opt, reg_opt
from extract_data import initialize_file, normalize, get_spectral_library, convert_library,remove_bands, prune_library
from skimage.morphology import square
from sklearn.preprocessing import Imputer
import spectral as spec
from segmentation import add_padding
spec.settings.WX_GL_DEPTH_SIZE = 16
bands = [0,1,103,104,105,106,107,108,109,110,111,112,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,220,221,222,223]
img = initialize_file('Cuprite.mat', key='X')
#img_shape = image.shape
img = img[425:614,86:335,:]
print('Image shape:', img.shape)
#view = spec.imshow(img, (100, 150, 180))
data = np.reshape(img,(img.shape[0]*img.shape[1],img.shape[2]))
data_imp = Imputer()
data_imp.fit(data)
data = data_imp.transform(data)
data = normalize(data)
dat_shape = data.shape
print('Data shape', dat_shape)
library = get_spectral_library('AVIRIS1995', 'Minerals')
library = prune_library(library,3)
M, names = convert_library(library)
dict_imp = Imputer()
dict_imp.fit(M)
M = dict_imp.transform(M)
M = normalize(M)
print('Number of samples in Dictionary',len(names))
M = remove_bands(M,bands)
data = remove_bands(data,bands)
recon_img = np.reshape(data.transpose(),(data.shape[1],img.shape[0],img.shape[1]))
#spec.view_cube(recon_img)
print('Dictionary Shape',M.shape)
print('Data reduced shape',data.shape)
pad_img = add_padding(recon_img,3)
print('Padded image', pad_img.shape)

M = M.transpose()
data = data.transpose()

mu = .004
lamb = .1
gamma = .006
n_iter = 300
width = 5
strel = square(width)
#data = np.transpose(data)

X = morph_opt(M,data,lamb,gamma,mu, strel, n_iter=300)
#M = np.transpose(M)
print(X.shape)
recon = np.dot(M,X)
#recon = np.transpose(recon)
recon_img = np.reshape(recon,(250,190,188))


