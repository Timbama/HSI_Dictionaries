import numpy as np
from extract_data import get_spectral_library, normalize, convert_library, create_sample, prune_library, initialize_file
from create_hsi import create_hsi 
from sklearn.preprocessing import Imputer
from sklearn.decomposition import sparse_encode
from skimage.morphology import opening, closing, erosion, dilation, square
import matplotlib.pyplot as plt
import matplotlib as mpl
def row_soft(X, tau):
    nu = np.sqrt(np.sum(np.power(X,2), axis=0))
    zero = np.zeros(nu.shape)
    A = np.maximum(zero,nu)
    A = np.divide(A,(np.add(A,tau)))
    A = np.reshape(A,(1,A.shape[0]))
    Y = np.tile(A,(X.shape[0],1))
    Y = np.multiply(Y,X)
    return Y
def comp_soft(X, tau):
    shape = X.shape
    Y = np.zeros(shape)
    Y = np.sign(X)*np.maximum((np.abs(X)-tau),0)
    Y = np.multiply(Y,X)
    return Y
mu = .02
lamb = .05
gamma = .05
n_iter = 1000
width = 4
strel = square(width)
M = initialize_file('USGS_pruned_10_deg.mat','B')
data = initialize_file('SNR 30 data.mat','Y')
actual = initialize_file('SNR 100 data.mat','Y')
actual = np.transpose(actual)
data = np.transpose(data)
Y = data
M = np.transpose(M)
print(M.shape)
sample = data[0:5,:]
image = create_hsi(sample)
print(image.shape)
Y = np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2]))
print("Dictionary shape:", M.shape)
X = sparse_encode(Y, M)

X = np.transpose(X)
M = np.transpose(M)
print("Dictionary shape:", M.shape)
Y = np.transpose(Y)
print('Reshaped data', Y.shape)
print('Encoded Shape', X.shape)
shape = X.shape
v1 = np.dot(M,X)
print('should be same shape as data: ', v1.shape)
v2 = X
v3 = X
v4 = X

d1 = np.zeros(v1.shape)
d2 = np.zeros(shape)
d3 = np.zeros(shape)
d4 = np.zeros(shape)
print(d4.shape)
i = 0
while i < 2000:
    x_hat = opening(X, strel)
    if i%10 == 0:
        v10 = v1
        v20 = v2
        v30 = v3
        v40 = v4
    I = np.identity(M.shape[1])
    MT = np.transpose(M)
    
    MTM= np.dot(MT,M)
    
    term_a = np.linalg.inv(MTM+(3*I))
    
    test = np.add(v1,d1)
    term_b = np.dot(MT,(np.add(v1,d1)))
    
    term_c = v2+d2
    term_d = v3+d3
    term_e = v4+d4
    X = np.dot(term_a,(term_b+term_c+term_d+term_e))
    v1 = (Y + mu*(np.dot(M,X)-d1))/(mu+1)
   
    v2 = row_soft((X-d2),lamb/mu)
   
    v3 = comp_soft((X-d3-x_hat),gamma/mu) + x_hat

    v4 = X - d4

    d1 = d1 - np.dot(M,X) + v1
    d2 = d2 - X + v2
    d3 = d3 - X + v3
    d4 = d4 - X + v4
    i+=1
    print(i)

recon = np.dot(M,X)
recon = np.transpose(recon)
#error = np.linalg.norm((actual-recon),'fro')
recon = np.reshape(recon, (75,75,224))
#print(error)
norm = mpl.colors.Normalize(vmin=0.,vmax=1.)
plt.subplot(211); plt.imshow(image[:,:,0]); plt.title('Origional')
plt.subplot(212); plt.imshow(recon[:,:,0]); plt.title('reconstruction')
plt.show()