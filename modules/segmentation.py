import numpy as np
from modules.extract_data import initialize_file, extract_pixel

def create_mask(data):
    data_pad = addPadding(data)
    mask = np.zeros(shape=(data.shape[0],data.shape[1]))
    mask_high = np.zeros(shape=(data.shape[0],data.shape[1]))
    mask_low = np.zeros(shape=(data.shape[0],data.shape[1]))
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            center = extract_pixel(x,y, data)
            window = np.zeros(shape=(data.shape[2],0))
            for z in range(3):
                for n in range(3):
                    window = np.hstack((window, extract_pixel(z+x,n+y,data_pad).transpose()))
            for f in range(window.shape[1]):
                mask[x][y] = np.add(mask[x,y], np.mean(np.linalg.norm(center.transpose().flatten() - window[:,f])))
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if(mask[x,y] < np.mean(mask)+8000):
                mask_high[x,y] = 0
            else:
                mask_high[x,y] = 1
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if(mask_high[x,y] == 1):
                mask_low[x,y] = 0
            else:
                mask_low[x,y] = 1
    return mask_high, mask_low
def addPadding(data):
    data_pad = np.empty((data.shape[0]+2, data.shape[1]+2, 0))
    for x in range(data.shape[2]):
        data_temp = np.pad(data[:,:,x], 1, 'mean')
        data_temp = np.expand_dims(data_temp, axis=0)
        data_temp = np.reshape(data_temp, (data_temp.shape[1], data_temp.shape[2], data_temp.shape[0]))
        data_pad = np.append(data_pad, data_temp, axis=2)
    return data_pad
def apply_mask(data, low_path, high_path):
    #Create a empty data sets to hold high and low structure data
    data_high = np.zeros(shape=(data.shape[0],data.shape[1],0))
    data_low = np.zeros(shape=(data.shape[0],data.shape[1],0))
    #Load the masks from file
    high_mask = np.load(high_path)
    low_mask = np.load(low_path)
    #apply the mask to the provided data
    for x in range(data.shape[2]):
        data_high = np.append(data_high, np.expand_dims(np.multiply(data[:,:,x], high_mask), axis=2), axis=2)
        data_low = np.append(data_low, np.expand_dims(np.multiply(data[:,:,x], low_mask), axis=2), axis=2)
    #reshape the data to match to a set of samples
    data_high = np.reshape(data_high, (data.shape[0]*data.shape[1], data.shape[2]), order='C')
    data_low = np.reshape(data_low, (data.shape[0]*data.shape[1], data.shape[2]), order='C')
    #find the the columns of the data that contains only zeros and remove them
    bad_cols_high = np.where(data_high.sum(axis=1) == 0)[0]
    data_high = np.delete(data_high, bad_cols_high, axis=0)
    bad_cols_low = np.where(data_low.sum(axis=1) == 0)[0]
    data_low = np.delete(data_low, bad_cols_low, axis=0)
    return data_high, data_low
