import scipy as sp
import numpy as np

timings = {'ha_START' = ??, 'ha_END' = ??, 'bit_START' = ??, 'bit_END' = ??, 'habit_START' = ??, 'habit_END' = ??, 'OVERLAP' = 100}

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


