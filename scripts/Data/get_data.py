import scipy as sp
from scipy import io
import numpy as np
import os
import pickle

PATH = '../../data_mat/'
SYMBOLS = ['HA', 'BIT', 'HABIT', 'SIG', 'NAL', 'SIGNAL']

# Input: string of file name
# Return: 2d array with row a channel and column as reading
def matFileTo2dMatrix(sample):
    return sp.io.loadmat(sample)['structData']

def storeSymbolAsPickle(className):
    all_samples = []

    CLASSPATH = PATH + className.upper() + '/'

    # Getting all MAT files
    # Neglecting the first file (.DS_Store)
    all_MAT_files = os.listdir(CLASSPATH)[1:]
    for MAT_file in all_MAT_files:
        sample_path = CLASSPATH + MAT_file
        sample_matrix = matFileTo2dMatrix(sample_path)
        all_samples.append(sample_matrix)

    symbols = np.array(all_samples)[:, np.newaxis, :, :]

    f = open('../../data/%s' % className.upper(), 'wb')
    pickle.dump(symbols, f)

for symbol in SYMBOLS:
	storeSymbolAsPickle(symbol)


