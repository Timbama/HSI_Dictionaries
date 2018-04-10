import numpy as np
from extract_data import get_spectral_library, normalize, convert_library, create_sample
from create_hsi import create_hsi
from sklearn.preprocessing import Imputer
from sklearn.decomposition import sparse_encode
from skimage.morphology import opening, closing, erosion, dilation, square
def row_soft(X, tau):
    nu = np.sqrt(np.sum(np.power(X,2), axis=0))
    print(nu.shape)
    zero = np.zeros(nu.shape)
    A = np.maximum(zero,nu)
    print(A.shape)
    A = np.divide(A,(np.add(A,tau)))
    A = np.reshape(A,(A.shape[0],1))
    print(A.shape)
    Y = np.tile(A,(1,X.shape[0]))
    print(Y.shape)
    Y = np.multiply(Y,np.transpose(X))
    return Y
def comp_soft(X, tau):
    shape = X.shape
    Y = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            Y[i,j] = np.sign(X[i,j])*np.maximum((np.abs(X[i,j]-tau)),0)
    print(Y.shape)
    return Y
K = 15
h = .025
mu = .5
lamb = .005
gamma = .005
n_iter = 1000
width = 2
strel = square(width)
library = get_spectral_library('AVIRIS2014','Minerals')
samples, names = create_sample(library)
samples = normalize(samples)

image = create_hsi(samples)
print(image.shape)

Y = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
print(Y.shape)

dictionary =  convert_library(library)
print(dictionary.shape)
imputer_data = Imputer()
imputer_dict = Imputer()
imputer_data.fit(Y)
imputer_dict.fit(dictionary)
data = imputer_data.transform(Y)
print(data.shape)
M = imputer_dict.transform(dictionary)
print(M.shape)

X = sparse_encode(data, M)
X = np.transpose(X)
print(X.shape)
shape = X.shape
M = np.transpose(M)
print(M.shape)
v1 = np.dot(M,X)
print(v1.shape)
v2 = X
v3 = X
v4 = X

d1 = np.zeros(shape)
d2 = np.zeros(shape)
d3 = np.zeros(shape)
d4 = np.zeros(v1.shape)
i = 0
while i < n_iter:
    x_hat = opening(X, strel)
    if i%10 == 0:
        v10 = v1
        v20 = v2
        v30 = v3
        v40 = v4
    I = np.identity(M.shape[1])
    MT = np.transpose(M)
    print(I.shape)
    MTM= np.dot(MT,M)
    print(MTM.shape)
    term_a = np.linalg.inv(MTM+(3*I))
    print(term_a.shape)
    term_b = np.dot(M.T,(v1+d1))
    print(term_b.shape)
    term_c = v2+d2
    term_d = v3+d3
    term_e = v4+d4
    X = np.dot(term_a,(term_b+term_c+term_d+term_e))
    v1 = (Y + mu*(np.dot(X,M)-d1))/(mu+1)
    print(v1.shape)
    v2 = row_soft((X-d2),lamb/mu)
    print(v2.shape)
    v3 = comp_soft((X-d3-x_hat),gamma/mu) + x_hat
    print(v3.shape)
    v4 = X - d4
    print(v4.shape)
    i+=1
print(X.shape)
