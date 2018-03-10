import numpy as np
import tifffile as tif
import csv
import random
from scipy import io
import os.path
def initialize_file(filename):
    extension = os.path.splitext(filename)[1]

    if extension == '.tiff':
        data = tif.imread(filename)
        return data
    elif extension == '.mat':
        name = os.path.splitext(filename)[0]
        name = os.path.basename(name)
        name = first_lower(name)
        mat_file = io.loadmat(filename)
        mat_file.keys()
        data = mat_file[name]
        data = np.array(data)
        return data
    else:
        data = np.empty(1)
        print("cannot read " + extension)
        return data
    

def extract_pixel(x, y, data):
    sample = np.empty((0, data.shape[2]))
    sample = np.append(sample, [data[x,y,:]], axis=0)
    return sample
def extract_pixel_random(num, data):
    sample = np.empty((0,data.shape[2]))
    for i in range(num):
        randX = random.randint(0,data.shape[0]-1)
        randY = random.randint(0,data.shape[1]-1)
        sample = np.append(sample, [data[randX,randY,:]], axis=0)
        i+=1
    return sample
def first_lower(s):
    if len(s) == 0:
      return s
    else:
      return s[0].lower() + s[1:]
