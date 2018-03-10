from scipy import io
import h5py
dataPath = 'G:/timba/Documents/Hyperspectral project/Data/'
mat_file = io.loadmat(dataPath + 'PaviaU.mat')
mat_file.keys()
print(mat_file.keys())
data_globals = mat_file['__globals__']
print(data_globals)
data_header = mat_file['__header__']
print(data_header)
data_version = mat_file['__version__']
data = mat_file['paviaU']
print(data)