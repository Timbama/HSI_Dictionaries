from time import time
from scipy import io
import h5py
import numpy as np
from extract_data import get_spectra
t0 = time()
d = get_spectra('Vegetation', 'AVIRIS2014' )

dt = time() - t0
print('done in %.2fs.' % dt)
print(d)
print(np.nanmax(d))
print(np.nanmin(d))
print(np.nanmean(d))
print(np.nanstd(d))
d = 6000*d
print(d)
print(np.nanmax(d))
print(np.nanmin(d))
print(np.nanmean(d))
print(np.nanstd(d))
