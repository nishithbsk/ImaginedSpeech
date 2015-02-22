import scipy
import sys
import os
import argparse
from data-interface import *

MIN_OVERLAP = 100

def euclideanScore(sample1, sample2):
    assert sample1.shape[1] == sample2.shape[1]
    return np.sum(np.linalg.norm(sample1 - sample2, axis=1))

def oneStrideAlignment(sample1, sample2, scorefn)
    sample1Size = sample1.shape[1]
    sample2Size = sample2.shape[1]
    scores = {}
    for i in range(-sample2Size+1, sample1Size):
        sample2left = 0 if i > 0 else -i
        sample2right = sample2Size-1 if i+sample2Size-1 < sample1Size else sample1Size-i
        sample1left = 0 if i < 0 else i
        sample1right = sample1Size if sample2right >= sample1Size-1
        score = scorefn(sample1[:, sample1left:sample1right], sample2[:, sample2left:sample2right])
        scores[i] = score
    return scores

#returns both preproccesed samples
def preprocess(sample, name):
    return truncate(sample, timings[name + "_START"], timings[name + "_END"])

def getSamples():
    parser = argparse.ArgumentParser(description='Performs sequence alignment on two classes of samples and returns an output file with average alignment scores')
    parser.add_argument('--path1', dest='path1', help='Path name with all the matlab files for first class of samples')
    parser.add_argument('--name1', dest='namt1', help="Unique name of first class in lower case. E.g. 'ba' or 'ha'")
    parser.add_argument('--path2', dest='path1', help='Path name with all the matlab files for second class of samples')
    parser.add_argument('--name2', dest='namt1', help="Unique name of second class in lower case. E.g. 'ba' or 'ha'")
    args = parser.parse_args()

    samples1 = []



    for file1 in os.listdir(args.path1)
        samples1.append(preprocess(matFileTo2dMatrix(file1)))

    os.listdir(args.path2)

sample1 = matFileTo2dMatrix(sys.argv[1])
sample2 = matFileTo2dMatrix(sys.argv[2])

samples = preprocess()

maxScore = oneStrideAlignment(sample1, sample2, euclideanScore)
