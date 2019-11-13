import numpy as np
from sklearn.decomposition import sparse_encode
from skimage.morphology import opening, closing, erosion, dilation
from segmentation import create_SAD_mat
from dictionary_update import update_dict
from sklearn.preprocessing import normalize
from evaluate import calculate_rsme
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
def create_nbd(Y, size, center):
    img = np.reshape(Y, (250,190,Y.shape[0]))
    sub_img = img[center-size:center+size,center-size:center+size, :]
    return sub_img
# def calc_SAD(img):
#     center = (np.floor(img.shape[0]/2.0), np.floor(img.shape[1]/2.0))
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if (i,j) != center:
def calc_var(img):
    var = img.var(axis=0)
    return var
def vector_morph(X, M, shape, depth=6):
    Y = np.dot(M,X)
    var = calc_var(Y)
    var_idx = np.flip(np.argsort(var), axis=-1)
    var_idx_cut = var_idx[:depth]
    img = np.reshape(Y, (depth, shape[0], shape[1]))
    create_SAD_mat()
def morph(X, strel, operation="opening"):
    if operation == "erosion":
        X_out = erosion(X)
    elif operation == "dilation":
        X_out = dilation(X)
    elif operation == "closing":
        X_out  = closing(X)
    else:
        X_out = opening(X)
    return X_out


def morph_opt(M, Y, lamb, gamma, mu, strel, operation, n_iter=2000, verbose=True):
    '''
    Args:
        M: dictionary of shape bands x atoms
        Y: data of shape bands x samples
        lamb: primal constant
        gamma: dual constant
        mu: regularization constant
        strel: structuring element
        n_iter: number of iterations
        verbose: determins verbosity
    Returns:
        This function returns a sparse matrix representation of the input dat with respect to the dictionary
    '''
    shape = (M.shape[1], Y.shape[1])
    #initilize data and sparse representation
    #X = sparse_encode(Y, M, max_iter=1)
    MT = np.transpose(M)
    MTM = np.dot(MT,M)
    IF = np.linalg.inv(MTM)
    inverse_max = np.max(IF)
    #X = np.random.rand(shape[0],shape[1])
    X = np.dot(np.dot(IF,MT),Y)
    X = normalize(X)
    X_max = np.max(X)
    print(X.shape)
    #initilize the seperable representations of X
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
    x_hat = erosion(X, strel)

    while i < n_iter:
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
        
        x_hat = morph(X, strel, operation)
        M = update_dict(M,Y,X)
        MT = np.transpose(M)
        MTM = np.dot(MT,M)
        if (i%10 == 0 and verbose):
            X_max = np.max(X)
            v1_max = np.max(v1)
            v2_max = np.max(v2)
            v3_max = np.max(v3)
            v4_max = np.max(v4)
            M_max = np.max(M)
            term_1 = np.linalg.norm(np.dot(M,X)-v1,)**2
            term_2 = np.linalg.norm(X-v2,)**2
            term_3 = np.linalg.norm(X-v3,)**2
            term_4 = np.linalg.norm(X-v4,)**2
            prime = np.sqrt(np.linalg.norm(np.dot(M,X)-v1,)**2 + np.linalg.norm(X-v2,)**2 + np.linalg.norm(X-v3,)**2 + np.linalg.norm(X-v4,)**2)
            dual = mu*np.linalg.norm(np.dot(np.transpose(M),(v1-v10))+v2-v20+v3-v30+v4-v40)
            rsme = calculate_rsme(Y,np.dot(M,X))
            print("iteration:" + str(i) + "\tprimal:" + str(prime) + "\tdual:" + str(dual) + "\trsme:" + str(rsme))
            if prime > 10*dual:
                mu = mu*2
                d1 = d1/2
                d2 = d2/2
                d3 = d3/2
                d4 = d4/2
            elif dual > 10*prime:
                mu = mu/2
                d1 = d1*2
        i+=1
    return X,M
def reg_opt(M, Y, lamb, mu, n_iter=2000, verbose=True):
    '''
    Args:
        M: dictionary of shape bands x atoms
        Y: data of shqape bands x samples
        lamb: primal constant
        gamma: dual constant
        mu: regularization constant
        n_iter: number of iterations
        verbose: determins verbosity
    Returns:
        This function returns a sparse matrix representation of the input data with respect to the dictionary

    '''
     #initilize data and sparse representation
    #X = sparse_encode(Y, M, max_iter=1)
    MT = np.transpose(M)
    MTM = np.dot(MT,M)
    IF = np.linalg.inv(MTM)
    X = np.dot(np.dot(IF,MT),Y)
    
    #initilize the seperable representations of X
    shape = X.shape
    v1 = X
    #Initilize the lagrangians to zero
    d1 = np.zeros(shape)
    Z = np.zeros(shape)
    i = 0
    #Initilize the parameters for the X update
    I = np.identity(M.shape[1])
    
    while i < n_iter:
        if i%10 == 0 and verbose:
            v10 = Z
        #update X
        term_a = np.linalg.inv(MTM)
        term_b = np.dot(MT,Y)
        term_c = mu*(Z+d1)
        #Update the seprable versions of X
        Z = comp_soft((X-d1),lamb/mu)
        X = np.dot(term_a,(term_b+term_c))
        #Update the Lagranians
        d1 = d1 - X + Z
        if i%10 == 0 and verbose:
            prime = np.sqrt(np.linalg.norm(X-Z,'fro')**2)
            dual = mu*np.linalg.norm((Z-v10),'fro')
            print("iteration:" + str(i) + "\tprimal:" + str(prime) + "\tdual:" + str(dual))
            if prime > 10*dual:
                mu = mu*2
                d1 = d1/2
            elif dual > 10*prime:
                mu = mu/2
                d1 = d1*2
        i+=1            
    return X
#def vec_opt(M, Y, lamb, gamma, mu, strel, n_iter=2000, verbose=True, depth=6):
  