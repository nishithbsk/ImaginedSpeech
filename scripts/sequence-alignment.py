import scipy
import sys
import os
import argparse
from data-interface import *

MIN_OVERLAP = 100

def euclideanScore(sample1, sample2):
    assert sample1.shape[1] == sample2.shape[1]
    return np.sum(np.linalg.norm(sample1 - sample2, axis=1))

INCLUDE OVERLAPPING
def oneStrideAlignment(sample1, sample2, scores, scorefn)
    sample1Size = sample1.shape[1]
    sample2Size = sample2.shape[1]
    for i in range(-sample2Size+1, sample1Size):
        sample2left = 0 if i > 0 else -i
        sample2right = sample2Size-1 if i+sample2Size-1 < sample1Size else sample1Size-i
        sample1left = 0 if i < 0 else i
        sample1right = sample1Size if sample2right >= sample1Size-1
        score = scorefn(sample1[:, sample1left:sample1right], sample2[:, sample2left:sample2right])
        scores[i] += score
    return scores

#returns both preproccesed samples
def preprocess(sample, name):
    return truncate(sample, timings[name + "_START"], timings[name + "_END"])

def parseArgs():
    parser = argparse.ArgumentParser(description='Performs sequence alignment on two classes of samples and returns an output file with average alignment scores')
    parser.add_argument('--composite', dest='comp', help='Unique class name of the composite sample')
    parser.add_argument('--constitutent', dest='const', help='Unique class name of the constitutent sample')
    parser.add_argument('--position', dest='position', help='Expected position of constituent in composite. "he" in "helium" for a syllable model would be 1/3')
    args = parser.parse_args()
    return parser.parse_args()

def getSamples(args):
    composites = getClassSamples(args.comp, preprocess)
    constituents = getClassSamples(args.const, preprocess)

    return (composites, constituents)

def getExpected(args):
    fraction = args.position
    return (int(fraction.split('/')[0]), int(fraction.split('/')[1]))

def pairwiseAlign(samples1, samples2):
    scores = {}
    for sample1 in samples1:
        for sample2 in samples2:
            oneStrideAlignment(sample1, sample2, scores, euclideanScore)
    for key in scores.keys():
        scores[key] /= len(samples1)*len(samples2)

    return scores

def analyze(averageScores, args)
    (expectedIndex, indicesSize) = getExpected(args)
    minAlignment =
    maxAlignment =

    increment = float(maxAlignment-minAlignment)/float(indicesSize)
    segments = [minAlignment + increment*i for i in range(indicesSize+1)]
    bestClassIndices =

    output = open("%s-%s.align" % (args.const, args.comp)
    print >>output, "Component: %s" % args.const
    print >>output, "Constituent: %s" % args.comp
    print >>output, "Expected Index: %d" % expectedIndex
    print >>output, "Number of Indices: %d" % indicesSize
    print >>output, ""
    for key in averageScores.keys():
        print >>output, "%d: %d" % averageScores[key]


def main():
    args = parseArgs()
    (composites, constituents) = getSamples(args)
    averageScores = pairwiseAlign(composites, constituents)
    evaluate(averageScores, args)

if __name__ == "__main__":
    main()
