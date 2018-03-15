from time import time
from scipy import io
import h5py
import numpy as np
from extract_data import initialize_file, addPadding, extract_pixel

dataPath = 'C:/Users/Timothy/OneDrive - Kennesaw State University/data/Data/'

file_name = dataPath + 'paviaU.mat'

data = initialize_file(file_name)
data_pad = addPadding(data)
print(data.shape, data_pad.shape)
window = np.zeros(shape=(data.shape[2],0))
result = np.zeros(shape=(data.shape[0],data.shape[1]))
for x in range(data.shape[0]):
    for y in range(data.shape[1]):
        center = extract_pixel(x,y, data)
        for z in range(x+1):
            for n in range(y+1):
               window = np.append(window, extract_pixel(z,n,data_pad))
        for f in range(window.shape[0]):
            result[x][y] += np.mean(np.linalg.norm(center - window[f]))
print(result.shape)            

    
