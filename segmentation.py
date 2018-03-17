import numpy as np
from extract_data import initialize_file, extract_pixel

def create_mask(data):
    data_pad = addPadding(data)
    mask = np.zeros(shape=(data.shape[0],data.shape[1]))
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
                mask[x,y] = 0
            else:
                mask[x,y] = 1
    return mask
def addPadding(data):
    data_pad = np.empty((data.shape[0]+2, data.shape[1]+2, 0))
    for x in range(data.shape[2]):
        data_temp = np.pad(data[:,:,x], 1, 'mean')
        data_temp = np.expand_dims(data_temp, axis=0)
        data_temp = np.reshape(data_temp, (data_temp.shape[1], data_temp.shape[2], data_temp.shape[0]))
        data_pad = np.append(data_pad, data_temp, axis=2)
    return data_pad