if __name__ == '__main__':
    # import required libraries
    import h5py as h5
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import gdal, osr
    import neon_aop_refl_hdf5_functions as neon
    # Read H5 file
    f = h5.File("NEON_D08_TALL_DP1_20170510_151641_reflectance.h5", "r")
    def list_dataset(name,node):
        if isinstance(node, h5.Dataset):
            print(name)
            print(node)

    #f.visititems(list_dataset)
    #Creates a varible to store the path to the relectance dataset
    tall_refl = f['TALL']['Reflectance']
    print(tall_refl)
    #creates an copy of the reflectance data
    tall_reflArray = tall_refl['Reflectance_Data']
    print(tall_reflArray)
    #stores shape information for reflectance data
    refl_shape = tall_reflArray.shape
    print('TALL Reflectence Data dimensins: ', refl_shape)
    #creates a path to the spectral characteristics
    wavelengths = tall_refl['Metadata']['Spectral_Data']['Wavelength']
    print(wavelengths)
    #Finds values of the minimum and maximum wavlength and the bandwidth
    min_wavelength = np.amin(wavelengths)
    min_wavelengthBand = wavelengths[1] - wavelengths[0]
    max_wavelengthBand = wavelengths[-1] - wavelengths[-2]
    max_wavelength = np.amax(wavelengths)
    print('Minimum Wavelength', min_wavelength, 'Bandwidth:', min_wavelengthBand, 'nm')
    print('Maximum Wavelength', max_wavelength, 'Bandwidth:', max_wavelengthBand, 'nm')
    #creats a path to the spatial information
    tall_mapInfo = tall_refl['Metadata']['Coordinate_System']['Map_Info']
    #prints the coordinate map information
    print('TALL Map Info', tall_mapInfo.value)
    mapInfo_string = str(tall_mapInfo.value)#convert map info to a string
    mapInfo_array = mapInfo_string.split(",")#Sepearte the string so it can be indexed
    print(mapInfo_array)
    #Extract resolution information and covert is to a floating point number
    res = float(mapInfo_array[5]),float(mapInfo_array[6])
    print('Resolution:', res)
    #Extract the maximum and minimum coordinate values for x and y
    xMin = float(mapInfo_array[3])
    xMax = xMin + (refl_shape[1]*res[0])
    yMax = float(mapInfo_array[4])
    yMin = yMax - (refl_shape[0]*res[1])
    #Create a tuple to hold the min and mak values
    tall_ext = (xMin, xMax, yMin, yMax)
    print('Tall ext', tall_ext)
    tall_extDict = {}
    tall_extDict['xMin'] = xMin
    tall_extDict['xMax'] = xMax
    tall_extDict['yMin'] = yMin
    tall_extDict['yMax'] = yMax
    print("Tall_dict: ", tall_extDict)
    #Extract a single band from the array
    b56 = tall_reflArray[:,:,55].astype(np.float)
    #show the type of the band extracted
    b56_type = type(b56)
    print('Band 56 Type:', b56_type)
    #show the shape of the band extracted
    b56_shape = b56.shape
    print('Band 56 Shape', b56_shape)
    #print out the unscaled raw reflectance data
    print('Band 56 Reflectance:\n', b56)
    #substitute the data ignore value (-9999)
    noDataValue = tall_reflArray.attrs['Data_Ignore_Value']
    b56[b56==int(noDataValue)] = np.nan
    #apply the scale factor
    scaleFactor = tall_reflArray.attrs['Scale_Factor']
    print('Scale Factor', scaleFactor)
    b56 = b56/scaleFactor
    #print out the cleaned band
    print('Cleaned Band 56 Reflectance: \n', b56)
    #plot the histogram of the extracted band
    """
    plt.hist(b56[~np.isnan(b56)],50)
    plt.title('Histogram of TALL Band 56 Reflectanc')
    plt.xlabel('Reflectance')
    plt.ylabel('Frequency')
    plt.show()
    ...
    """
    #Plot  the Relectance of band 56
    """
    tall_fig = plt.figure(figsize=(18,9))
    ax1 = tall_fig.add_subplot(1, 2, 1)
    tall_plot = ax1.imshow(b56, extent=tall_ext, cmap = 'jet')
    cbar = plt.colorbar(tall_plot, aspect=50)
    cbar.set_label('Reflectance')
    plt.title('TALL Band 56 Reflectance')
    ax1.ticklabel_format(useOffset=False, style='plain')
    rotatexlabel = plt.setp(ax1.get_xticklabels(), rotation=270)
    tall_plot
    plt.show()
    """
    
