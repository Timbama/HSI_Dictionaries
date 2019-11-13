
import numpy as np
from numpy import linalg as LA
def compare_abundance(gt, exp):
    diff = np.subtract(gt, exp)
    print("Shape of Image", diff.shape)
    norm = LA.norm(diff, axis=0)
    print("Shape of Norm", norm.shape)
    avg = np.sqrt(np.sum(norm)/(gt.shape[0]*gt.shape[1]))
    return avg
def calculate_rsme(gt, exp):
    rsme_square = np.divide(np.sum(np.square(np.subtract(gt,exp))),gt.shape[1])
    rsme = np.sqrt(rsme_square)
    return rsme