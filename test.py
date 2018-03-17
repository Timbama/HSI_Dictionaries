from time import time
import numpy as np
from segmentation import create_mask
from extract_data import initialize_file
import matplotlib.pyplot as plt

dataPath = 'G:/timba/Documents/Hyperspectral project/Data/'

file_name = dataPath + 'paviaU.mat'

data = initialize_file(file_name)

t0 = time()
mask = create_mask(data)
dt = time() - t0
print('done in %.2fs.' % dt)

np.save('mask', mask)
plt.imshow(mask)
plt.show()    