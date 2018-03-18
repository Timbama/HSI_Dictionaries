from time import time
import numpy as np
from modules.segmentation import create_mask, apply_mask
from modules.extract_data import initialize_file
import matplotlib.pyplot as plt

dataPath = 'G:/timba/Documents/Hyperspectral project/Data/'

file_name = dataPath + 'paviaU.mat'
low_path = 'low_mask.npy'
high_path = 'high_mask.npy'
data = initialize_file(file_name)

print('Creating a mask')
t0 = time()
high_mask, low_mask = create_mask(data)
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