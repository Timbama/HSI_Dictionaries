import numpy as np
import tifffile as tif
import csv
import random
from scipy.io import loadmat
import os.path
from USG_data_paths import dataPath, sensor_type, spectra_types
def initialize_file(filename):
    extension = os.path.splitext(filename)[1]
    if extension == '.tiff':
        data = tif.imread(filename)
        return data
    elif extension == '.mat':
        name = os.path.splitext(filename)[0]
        name = os.path.basename(name)
        name = first_lower(name)
        mat_file = loadmat(filename)
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
def get_spectra(types, sensor):
    path = dataPath + sensor_type[sensor]  + '/' + spectra_types[types] + '/'
    directory = os.fsencode(path)
    print(path + os.listdir(directory)[0].decode())
    f = open(path + os.listdir(directory)[0].decode())
    size = len(f.readlines())
    data = np.array([]).reshape(0,size-1)
    for file in os.listdir(directory):
        i = 0
        temp = []
        print(file.decode())
        f = open(path + file.decode(), 'r')
        for line in f:
            if(i==0):
                i+=1
            else:
                numline = float(line)
                if(numline == -1.23e34):
                    numline = np.nan
                numline = numline*6000
                temp.append(numline)
        temp  = np.array(temp)
        data = np.vstack([data, temp])  
    return data 
def get_spectra_no_dup(sensor, types):
    path = dataPath + sensor_type[sensor]  + '/' + spectra_types[types] + '/'
    directory = os.fsencode(path)
    init = open(path + os.listdir(directory)[0].decode())
    size = len(init.readlines())
    data = np.array([]).reshape(0,size-1)
    first = True
    prev_mat = split_at(init.name,'_', 3)
    for file in os.listdir(directory):
        i = 0
        temp = []
        f = open(path + file.decode(), 'r')
        if prev_mat != split_at(f.name,'_', 5) or first:
            for line in f:
                if(i==0):
                    i+=1
                else:
                    numline = float(line)
                    if(numline == -1.23e34):
                        numline = np.nan
                    numline = numline*6000
                    temp.append(numline)
            temp  = np.array(temp)
            data = np.vstack([data, temp])
            prev_mat = split_at(f.name,'_', 5)
            print(prev_mat)
            first = False
    return data
def get_spectral_library(sensor, types):
    path = dataPath + sensor_type[sensor]  + '/' + spectra_types[types] + '/'
    directory = os.fsencode(path)
    init = open(path + os.listdir(directory)[0].decode())
    first = True
    prev_mat = split_at(init.name,'_', 3)
    data = {}
    for file in os.listdir(directory):
        i = 0
        temp = []
        f = open(path + file.decode(), 'r')
        if prev_mat != split_at(f.name,'_', 5) or first:
            for line in f:
                if(i==0):
                    i+=1
                else:
                    numline = float(line)
                    if(numline == -1.23e34):
                        numline = np.nan
                    temp.append(numline)
            temp  = np.array(temp)
            data[prev_mat] = temp
            prev_mat = split_at(f.name,'_', 5)
            print(prev_mat)
            first = False
    return data
def split_at(input, delimiter, n):
    words = input.split(delimiter)
    data = words[n]
    return data
def create_sample(library, sensor, types):
    library = get_spectral_library(sensor, types)
    names = []
    names.append(random.sample(list(library.keys()), 5))
    names = names[0]
    print(names)
    size = 224
    samples = np.array([]).reshape(0,size)
    print(samples.shape)
    for n in names:
        temp = library[n]
        samples = np.vstack([samples, temp])
    return samples