
from time import time
import numpy as np
from segmentation import create_SAD_mat, apply_mask, addPadding, threshold_mask
from extract_data import initialize_file
import matplotlib.pyplot as plt

dataPath = 'G:/timba/Documents/Hyperspectral project/Data/Hyperspectral Images/'

file_name = dataPath + 'paviaU.mat'
low_path = 'low_mask.npy'
high_path = 'high_mask.npy'
data = initialize_file(file_name)
print('Add padding...')
t0 = time()
data_pad = addPadding(data)
dt = time() - t0
print('done in %.2fs.' % dt)

print('Creating a matrix of variations')
t0 = time()
mask = create_SAD_mat(data_pad)
dt = time() - t0
print('done in %.2fs.' % dt)
np.save('SAD_matrix', mask)
print(mask.shape)

print('Creating mask based on threshold value')
t0 = time()
high_mask, low_mask = threshold_mask(mask)
dt = time() - t0
print('done in %.2fs.' % dt)
np.save('low_mask', low_mask)
np.save('high_mask', high_mask)

print('Applying mask')
t0 = time()
high_data, low_data = apply_mask(data, low_path, high_path)
np.save('low_data', low_data)
np.save('high_data', high_data)
dt = time() - t0
print('done in %.2fs.' % dt)
plt.imshow(low_mask)
plt.show()