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
def morph_opt(M, Y, lamb, gamma, mu, strel, n_iter=2000, verbose=True):
    #initilize data and sparse representation
    X = sparse_encode(Y, M, max_iter=1)
    X = np.transpose(X)
    M = np.transpose(M)
    Y = np.transpose(Y)
    #initilize the seperable representations of X
    shape = X.shape
    v1 = np.dot(M,X)
    v2 = X
    v3 = X
    v4 = X
    #Initilize the lagrangians to zero
    d1 = np.zeros(v1.shape)
    d2 = np.zeros(shape)
    d3 = np.zeros(shape)
    d4 = np.zeros(shape)
    i = 0
    #Initilize the parameters for the X update
    I = np.identity(M.shape[1])
    MT = np.transpose(M)
    MTM= np.dot(MT,M)
    while i < n_iter:
        x_hat = opening(X, strel)
        if i%10 == 0 and verbose:
            v10 = v1
            v20 = v2
            v30 = v3
            v40 = v4
        #update X
        term_a = np.linalg.inv(MTM+(3*I))
        term_b = np.dot(MT,(np.add(v1,d1)))
        term_c = v2+d2
        term_d = v3+d3
        term_e = v4+d4
        X = np.dot(term_a,(term_b+term_c+term_d+term_e))
        #Update the seprable versions of X
        v1 = (Y + mu*(np.dot(M,X)-d1))/(mu+1)
        v2 = row_soft((X-d2),lamb/mu)
        v3 = comp_soft((X-d3-x_hat),gamma/mu) + x_hat
        v4 = X - d4
        #Update the Lagranians
        d1 = d1 - np.dot(M,X) + v1
        d2 = d2 - X + v2
        d3 = d3 - X + v3
        d4 = d4 - X + v4
        i+=1
        if i%10 == 0 and verbose:
            prime = np.sqrt(np.linalg.norm(np.dot(M,X)-v1,'fro')**2 + np.linalg.norm(X-v2,'fro')**2 + np.linalg.norm(X-v3,'fro')**2 + np.linalg.norm(X-v4,'fro')**2)
            dual = mu*np.linalg.norm(np.dot(np.transpose(M),(v1-v10))+v2-v20+v3-v30+v4-v40,'fro')
            print("iteration:" + str(i) + "\tprimal:" + str(prime) + "\tdual:" + str(dual))
    return X
mu = .08
lamb = .2
gamma = .08
n_iter = 1000
width = 3
strel = square(width)
M = initialize_file('USGS_pruned_10_deg.mat','B')
data = initialize_file('SNR 20 data.mat','Y')
actual = initialize_file('SNR 100 data.mat','Y')
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

X = morph_opt(M, Y, lamb, gamma, mu, strel, n_iter=5000)
M = np.transpose(M)
recon = np.dot(M,X)
recon = np.transpose(recon)
error = np.linalg.norm((actual-recon),'fro')
diff =np.abs(recon-actual)
avg_error = np.mean(diff)
max_error = np.amax(diff)
std_error = np.std(diff)
#recon = np.reshape(recon, (75,75,224))
print("Norm of the difference", error)
print("average error", avg_error)
print("Standad deviation of error", std_error)
print("max error", max_error)
plt.subplot(311); plt.imshow(actual); plt.title('Origional')
plt.subplot(312); plt.imshow(recon); plt.title('reconstruction')
plt.subplot(313); plt.imshow(diff); plt.title('diffrenece')
plt.show()