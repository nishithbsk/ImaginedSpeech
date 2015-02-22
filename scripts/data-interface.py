import scipy as sp
import numpy as np
import os

timings = {'ha_START' = ??, 'ha_END' = ??, 'bit_START' = ??, 'bit_END' = ??, 'habit_START' = ??, 'habit_END' = ??, 'OVERLAP' = 100}

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

def meanFeatureExtractor(sample):

# Do not change size of sample. No pooling.
def convNetActivation(sample):

def removeChannel(sample, channel):

def discreteWaveletTransform(sample):

def getNumberChannels(sample):

def getNumberReadings(sample):

def removeOverlap(sample):

def getClassSamples(className, preprocessfn = None):
	# array = whatever
	# if preprocessfn:
	# 	return map(lambda x: preprocessfn(x), array)
	# else:
	# 	return array
	# return preprocessed array of npmatrices of all class samples
	# return array of npmatrices of all class samples
	all_samples = []
	# Getting all MAT files
	# Neglecting the first file (.DS_Store)
	CLASSPATH = eval(className + 'PATH')

	all_MAT_files = os.listdir(CLASSPATH)[1:]
	for MAT_file in all_MAT_files:
		sample_path = CLASSPATH + MAT_file
		sample_matrix = matFileTo2dMatrix(sample_path)
		if preprocessfn:
			sample_matrix = preprocessfn(sample_matrix)
		all_samples.append(sample_matrix)

	return np.array(all_samples)






