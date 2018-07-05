import numpy as np
from USG_data_paths import two_member, three_member, four_member
def linear_mix(pixels, fracton=None):
    if fracton==None:
        fraction = 1.0/(pixels.shape[0])
    result = np.zeros((1,pixels.shape[1]))
    for i in range(pixels.shape[0]):
        result += fraction*pixels[i,:]
    return result
def construct_cube(pixel, upper_corner, shape, size):
    image = np.zeros((shape[0],shape[1], pixel.shape[1]))
    for i in range(size):
        for j in range(size):
            image[i + upper_corner[0], j + upper_corner[1],:] = pixel[0,:]
    return image
def create_hsi(samples):
    test_image = np.zeros((75,75,samples.shape[1]))
    n_samples = samples.shape[1]
    upper_corner = [5,5]
    shape = (75,75)
    for i in range(5):
        i += 1
        temp = np.zeros((0,samples.shape[1]))
        if i == 1:
            for j in range(n_samples):
                pixel = samples[j,:]
                pixel = np.reshape(pixel, (1, samples.shape[1]))
                cube = construct_cube(pixel, upper_corner, shape, 5)
                test_image += cube
                upper_corner[0] += 15 
        elif i == 2:
            for j in range(n_samples):
                temp = np.zeros((0,samples.shape[1]))
                for k in range(len(two_member[j])):
                    temp = np.vstack((temp, samples[two_member[j][k],:])) 
                temp = linear_mix(temp)
                cube = construct_cube(temp, upper_corner, shape, 5)
                test_image += cube
                upper_corner[0] += 15
        elif i == 3:  
            for j in range(n_samples):
                temp = np.zeros((0,samples.shape[1]))
                for k in range(len(three_member[j])):
                    temp = np.vstack((temp, samples[three_member[j][k],:])) 
                temp = linear_mix(temp)
                cube = construct_cube(temp, upper_corner, shape, 5)  
                test_image += cube
                upper_corner[0] += 15 
        elif i == 4: 
            for j in range(n_samples):
                temp = np.zeros((0,samples.shape[1]))
                for k in range(len(four_member[j])):
                    temp = np.vstack((temp, samples[four_member[j][k],:])) 
                temp = linear_mix(temp)
                cube = construct_cube(temp, upper_corner, shape, 5)  
                test_image += cube
                upper_corner[0] += 15
        else:
            for j in range(n_samples):
                temp = samples
                temp = linear_mix(temp)
                cube = construct_cube(temp, upper_corner, shape, 5) 
                test_image += cube
                upper_corner[0] += 15 
        upper_corner[0] = 5
        upper_corner[1] += 15 
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.sum(test_image[i,j,:]) == 0:
                test_image[i,j,:] = linear_mix(samples)
    test_image = np.swapaxes(test_image, 0,1)
    return test_image
def create_square(weight, shape, upper_corner, size):
    square = np.zeros(shape)
    for i in range(size):
        for j in range(size):
            square[i + upper_corner[0], j + upper_corner[1]] = weight
    return square
def abundance_map(weights, number, shape):
    upper_corner = [5,5]
    image = np.zeros((shape[0], shape[1]))
    start = (number)*15 + 5
    image += create_square(1, shape, (start, 5), 5)
    upper_corner[1] += 15
    for i in range(len(two_member)):
        for j in range(len(two_member[i])):
            upper_corner[0] = 5
            if number == two_member[i][j]:
                upper_corner[0] += i*15
                image += create_square(weights[0], shape, upper_corner, 5)
    upper_corner[1] += 15
    for i in range(len(three_member)):
        for j in range(len(three_member[i])):
            upper_corner[0] = 5
            if number == three_member[i][j]:
                upper_corner[0] += i*15
                image += create_square(weights[1], shape, upper_corner, 5)
    upper_corner[1] += 15
    for i in range(len(four_member)):
        for j in range(len(four_member[i])):
            upper_corner[0] = 5
            if number == four_member[i][j]:
                upper_corner[0] += i*15
                image += create_square(weights[2], shape, upper_corner, 5)
    upper_corner[1] += 15
    for i in range(5):
        upper_corner[0] = 5 + i*15
        image += create_square(weights[3], shape, upper_corner, 5)        
    image = np.swapaxes(image, 0,1)
    return image