import scipy as sp
import numpy as np
import os
import math

timings = {'HA_START' = ??, 'HA_END' = ??, 'BIT_START' = ??, 'BIT_END' = ??, 'HABIT_START' = ??, 'HABIT_END' = ??, 'OVERLAP' = 100}

HA_PATH = '../data/syllables/ha/'
BIT_PATH = '../data/syllables/bit/'
HABIT_PATH = '../data/syllables/habit/'
SIG_PATH = '../data/syllables/sig/'
NAL_PATH = '../data/syllables/nal/'
SIGNAL_PATH = '../data/syllables/signal/'

# Input: string of file name
# Return: 2d array with row a channel and column as reading
def matFileTo2dMatrix(sample):
	return sp.io.loadmat(sample)['structData']

def truncate(sample, left, right):
	return sample[left:right]

def meanFeatureExtractor(sample, num_parts):
	num_channels = getNumberChannels(sample)
	num_readings = getNumberReadings(sample)

	parts_size = math.floor(num_readings/num_parts)
	parts_array = zeros(num_channels, num_parts);

	for i in xrange(numParts):
		parts_array(:, i) = np.mean(sample(:, i * parts_size : (i + 1) * parts_size), axis=1)

	return reshape(parts_array, (1, num_parts * num_channels))


# Do not change size of sample. No pooling.
def convNetActivation(sample):

def removeChannel(sample, channel):
	return np.delete(sample, (channel), axis=0)

def discreteWaveletTransform(sample):

def getNumberChannels(sample):
	return sample.shape[0]

def getNumberReadings(sample):
	return sample.shape[1]

def getClassSamples(className, preprocessfn = None):
	all_samples = []
	
	CLASSPATH = eval(className.upper() + '_PATH')

	# Getting all MAT files
	# Neglecting the first file (.DS_Store)
	all_MAT_files = os.listdir(CLASSPATH)[1:]
	for MAT_file in all_MAT_files:
		sample_path = CLASSPATH + MAT_file
		sample_matrix = matFileTo2dMatrix(sample_path)
		if preprocessfn:
			sample_matrix = preprocessfn(sample_matrix, className)
		all_samples.append(sample_matrix)

	return np.array(all_samples)






