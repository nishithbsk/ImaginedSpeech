import pickle
import numpy

PATH = '../../data/'

def get_all_instances_of_symbol(symbol):
	f = open(PATH + symbol.upper(), 'rb')
	return pickle.load(f)