import os
import glob
import numpy as np

a = np.load('pd.npy')
b = np.load('npmap.npy')

print(np.allclose(a,b))