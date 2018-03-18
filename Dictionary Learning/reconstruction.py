from time import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Preprocessing.extract_data as ext

#Setup the pixels as data frames and plot the spectrums for comparison
"""
pixel_df = pd.DataFrame(index=wavelengths)
pixel_df['Actual'] = actual_pixel[0]
pixel_df['Reconstruction'] = recovered_pixel[0]
plt.figure(); pixel_df.plot(); plt.legend(loc='best')
plt.title('Spectral Signature for TALL pixel (5000,500)')
ax = plt.gca()
ax.set_xlim([np.min(wavelengths),np.max(wavelengths)])
ax.set_ylim([np.min(pixel_df['Actual']), np.max(pixel_df['Actual'])])
ax.set_xlabel("Wavelength, nm"); ax.set_ylabel("Reflectance")
ax.grid('on')
plt.show()
"""