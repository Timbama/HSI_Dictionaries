from time import time
import numpy as np
import modules.extract_data as ext
import modules.sparse_code as sparse
from sklearn.preprocessing import Imputer


train_pixels = np.load('low_data.npy')
n_components = 150

print('Find Sparse Representation for dictionary')
t0 = time()
dictionary = sparse.initialize_dict(n_components, train_pixels.shape[1])

code_mat = np.array([]).reshape(0, n_components)

for x in range(train_pixels.shape[0]):
    #print('sparsifying sample ', x+1, ' for dictionary')
    code = sparse.omp_sparse(dictionary, train_pixels[x,:].reshape(1, train_pixels.shape[1]))
    code_mat = np.vstack([code_mat, code])
dt = time() - t0
print('done in %.2fs.' % dt)
print(code_mat.shape)
print(np.count_nonzero(code_mat))
bad_cols = np.where(code_mat.sum(axis=1) == 0)[0]
print(bad_cols)
code_mat = np.delete(code_mat, bad_cols, axis=0)
print(code_mat.shape)
np.save('sparse_coded_data', code_mat)