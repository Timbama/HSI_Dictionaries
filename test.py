import numpy as np
from USGS_data_path_Laptop import dataPath_HSI
import matplotlib.pyplot as plt
from time import time
n_samples = 1000
print('Loading, normalizing and selecting training data')
t0 = time()
#load data
data = np.load(dataPath_HSI  + 'high_data.npy')
#normalize data set
data_norm = (data-np.amin(data))/(np.amax(data)-np.amin(data))
#select data to train dictionary
random_index = np.random.choice(data_norm.shape[0],size=n_samples,replace=False)
train_data  = data_norm[random_index,:]
dt = time() - t0
print('done in %.2fs.' % dt)
