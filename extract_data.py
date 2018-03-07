import numpy as np
import tifffile as tif
import csv
import random
def initialize_tiff(refl_filename, wave_filename):
    data = tif.imread(refl_filename)
    wavelengths = np.zeros(200)
    dataFile = open(wave_filename, 'r')
    temp = [list(map(float,rec)) for rec in csv.reader(dataFile, delimiter=',')]
    wavelengths = temp[0]   
    return data, wavelengths
def extract_pixel(x, y, data):
    sample = np.empty((0,220))
    sample = np.append(sample, [data[:,x,y]], axis=0)
    return sample
def extract_pixel_random(num, dim, data):
    sample = np.empty((0,220))
    for i in range(num):
        randX = random.randint(0,dim[0]-1)
        randY = random.randint(0,dim[1]-1)
        sample = np.append(sample, [data[:,randX,randY]], axis=0)
        i+=1
    return sample
