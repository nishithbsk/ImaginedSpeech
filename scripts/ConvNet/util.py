import pickle
import numpy

PATH = '../../data/'

def get_all_instances_of_symbol(symbol):
	f = open(PATH + symbol.upper(), 'rb')
	return pickle.load(f)

def plot(loss_history, train_acc_history, val_acc_history):
	plt.subplot(2, 1, 1)
	plt.plot(train_acc_history)
	plt.plot(val_acc_history)
	plt.title('accuracy vs time')
	plt.legend(['train', 'val'], loc=4)
	plt.xlabel('epoch')
	plt.ylabel('classification accuracy')

	plt.subplot(2, 1, 2)
	plt.plot(loss_history)
	plt.title('loss vs time')
	plt.xlabel('iteration')
	plt.ylabel('loss')
	plt.show()