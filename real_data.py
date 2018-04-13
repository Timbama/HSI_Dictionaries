import numpy as np
from Optimization import morph_opt, reg_opt
from extract_data import initialize_file, normalize, get_spectral_library, convert_library,remove_bands
from skimage.morphology import square
from sklearn.preprocessing import Imputer
bands = [0,1,103,104,105,106,107,108,109,110,111,112,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,220,221,222,223]
data = initialize_file('CupriteS1_F224.mat', key='Y')
#img_shape = image.shape
print('Image shape:', data.shape)
#data = np.reshape(image, (img_shape[0]*img_shape[1], img_shape[2]))
data_imp = Imputer()
data_imp.fit(data)
data = data_imp.transform(data)
data = normalize(data)
dat_shape = data.shape
print('Data shape', dat_shape)
library = get_spectral_library('AVIRIS1995', 'Minerals')
M, names = convert_library(library)
dict_imp = Imputer()
dict_imp.fit(M)
M = dict_imp.transform(M)
M = normalize(M)
M = np.transpose(M)
print('Number of samples in Dictionary',len(names))
M = remove_bands(M,bands)
data = remove_bands(data,bands)
print('Dictionary Shape',M.shape)
print('Data reduced shape',data.shape)

mu = .004
lamb = .1
gamma = .006
n_iter = 1000
width = 5
strel = square(width)

M = np.transpose(M)
data = np.transpose(data)

X = morph_opt(M,data,lamb,gamma,mu,strel)
M = np.transpose(M)
print(X.shape)
recon = np.dot(M,X)
recon = np.transpose(recon)
recon_img = np.reshape(recon,(250,190,dat_shape[0]))
