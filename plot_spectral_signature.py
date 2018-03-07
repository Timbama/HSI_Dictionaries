if __name__ == '__main__':
    # import required libraries
    import h5py as h5
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import gdal, osr
    import neon_aop_refl_hdf5_functions as neon
    
    #Create a subset of the dataset
    #create a dictionary that matches the extent to the locations in the data set
    tallRefl, tall_refl_md, wavelengths = neon.h5refl2array('NEON_D08_TALL_DP1_20170510_151641_reflectance.h5')
    for item in sorted(tall_refl_md):
        print(item + ':', tall_refl_md[item])
    ext_dict = tall_refl_md['ext_dict']
    xMin = ext_dict['xMin']
    xMax = ext_dict['xMax']
    yMin = ext_dict['yMin']
    yMax = ext_dict['yMax']
    clipExtDict = {}
    clipExtDict['xMin'] = xMin + 200.
    clipExtDict['xMax'] = xMax - 220.
    clipExtDict['yMin'] = yMin + 6000.
    clipExtDict['yMax'] = yMax - 6600.
    #create an extent from the dictionary
    clipExtent = (clipExtDict['xMin'],clipExtDict['xMax'],clipExtDict['yMin'],clipExtDict['yMax'])
    clipIndex = neon.calc_clip_index(clipExtDict, tall_refl_md['ext_dict'])

    tall_b56_subset = neon.subset_clean_band(tallRefl, tall_refl_md, clipIndex, 55)
    print('TALL Subset Stats')
    print('min: ', np.nanmin(tall_b56_subset))
    print('max: ', round(np.nanmax(tall_b56_subset),2))
    print('mean: ', round(np.nanmean(tall_b56_subset),2))

    neon.plot_band_array(tall_b56_subset, clipExtent, (0,.15), title='TALL subselt Band 56', cmap_title='Reflectance', colormap='gist_earth')
    #plt.show()
    print('TALL subset shape', tallRefl.shape)
    tall_pixel_df = pd.DataFrame()
    tall_pixel_df['reflectance'] = tallRefl[5000,500,:]/tall_refl_md['scaleFactor']
    tall_pixel_df['wavelengths'] = wavelengths
    print(tall_pixel_df.head(5))
    print(tall_pixel_df.tail(5))
    tall_pixel_df.plot(x='wavelengths', y='reflectance', kind='scatter', edgecolor='none')
    plt.title('Spectral Signature for TALL pixel (5000,500)')
    ax = plt.gca()
    ax.set_xlim([np.min(wavelengths),np.max(wavelengths)])
    ax.set_ylim([np.min(tall_pixel_df['reflectance']), np.max(tall_pixel_df['reflectance'])])
    ax.set_xlabel("Wavelength, nm"); ax.set_ylabel("Reflectance")
    ax.grid('on')
    bad_band_window1 = tall_refl_md['bad_band_window1']
    bad_band_window2 = tall_refl_md['bad_band_window2']
    plt.plot((bad_band_window1[0], bad_band_window1[0]), (0, 1.5), 'r--')
    plt.plot((bad_band_window1[1], bad_band_window1[1]), (0, 1.5), 'r--')
    plt.plot((bad_band_window2[0], bad_band_window2[0]), (0, 1.5), 'r--')
    plt.plot((bad_band_window2[1], bad_band_window2[1]), (0, 1.5), 'r--')
    #plt.show()
    #remove the bad band data
    #w = copy.copy(wavelengths.value)
   # w[((w >= bad_band_window1[0]) & (w <= bad_band_window1[1])) | ((w >= bad_band_window2[0]) & (w <= bad_band_window2[1]))] = np.nan
    #w[-10:] = np.nan
   # print(w)
