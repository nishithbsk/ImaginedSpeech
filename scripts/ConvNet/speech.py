import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *
from util import *

from classifiers.convnet import *
from MultiLevelConvNet import MultiLevelConvNet

X_sig = get_all_instances_of_symbol('SIG')[:50, :, :, 800:2201]
y_sig_1 = np.tile(np.arange(10), (X_sig.shape[0], 1))
y_sig_2 = np.tile(np.arange(2), (X_sig.shape[0], 1))
y_sig_3 = np.tile(0, (X_sig.shape[0], 1))

X_nal = get_all_instances_of_symbol('NAL')[:50, :, :, 800:2201]
y_nal_1 = np.tile(np.arange(10, 20), (X_nal.shape[0], 1))
y_nal_2 = np.tile(np.arange(2, 4), (X_nal.shape[0], 1))
y_nal_3 = np.tile(1, (X_nal.shape[0], 1))

order = np.random.permutation(X_sig.shape[0])

X = np.concatenate(X_sig, X_nal)[order]
y_1 = np.concatenate(y_sig_1, y_nal_1)[order]
y_2 = np.concatenate(y_sig_2, y_nal_2)[order]
y_3 = np.concatenate(y_sig_3, y_nal_3)[order]

X_train = X[:9*X.shape[0]/10]
X_val = X[9*X.shape[0]/10:]
y_1_train = y_1[:9*y_1.shape[0]/10]
y_1_val = y_1[9*y_1.shape[0]/10:]
y_2_train = y_2[:9*y_2.shape[0]/10]
y_2_val = y_2[9*y_2.shape[0]/10:]
y_3_train = y_3[:9*y_3.shape[0]/10]
y_3_val = y_3[9*y_3.shape[0]/10:]

component_dim = (1, 64, 140)

#############################################################################################################################################

print "Initializing layer models"

fn1 = speech_convnet
num_components_per_img1 = 1
input_component_dim1 = (component_dim[0], component_dim[1], component_dim[2] * num_components_per_img1)
model1 = init_speech_convnet(input_shape = input_component_dim1, 
							num_classes = 10, filter_size = 3, num_filters = (32, 128, 256), weight_scale = 1e-4)
output = fn1(X[:1, :input_component_dim1[0], :input_component_dim1[1], :input_component_dim1[2]], model1, extract_features = True)[0]
output_component_dim1 = output.shape
stride1 = input_component_dim1[2]

fn2 = speech_convnet
num_components_per_img2 = 5
input_component_dim2 = (output_component_dim1[0], output_component_dim1[1], output_component_dim1[2] * num_components_per_img2)
model2 = init_speech_convnet(input_shape = input_component_dim2, 
							num_classes = 2, filter_size = 3, num_filters = (32, 128, 256), weight_scale = 1e-4)
output = np.tile(output, (1, num_components_per_img2))
output = output[np.newaxis, :, :, :]
output = fn2(output[:1, :input_component_dim2[0], :input_component_dim2[1], :input_component_dim2[2]], model2, extract_features = True)[0]
output_component_dim2 = output.shape
stride2 = input_component_dim2[2]

fn3 = speech_convnet
num_components_per_img3 = 2
input_component_dim3 = (output_component_dim2[0], output_component_dim2[1], output_component_dim2[2] * num_components_per_img3)
model3 = init_speech_convnet(input_shape = input_component_dim3, 
							num_classes = 2, filter_size = 3, num_filters = (32, 128, 256), weight_scale = 1e-4)
output = np.tile(output, (1, num_components_per_img3))
output = output[np.newaxis, :, :, :]
output_component_dim3 = fn3(output[:1, :input_component_dim3[0], :input_component_dim3[1], :input_component_dim3[2]], model3, extract_features = True)[0].shape
stride3 = input_component_dim3[2]

print "Finished initializing layers"

##############################################################################################################################################

net = MultiLevelConvNet(3)
net.set_level_parameters(0, fn1, model1, input_component_dim1, num_components_per_img1, stride1)
net.set_level_parameters(1, fn2, model2, input_component_dim2, num_components_per_img2, stride2)
net.set_level_parameters(2, fn3, model3, input_component_dim3, num_components_per_img3, stride3)
net.set_level_learning_parameters(0, reg = 0.0000, learning_rate = 0.0015, batch_size = 250, num_epochs = 5, 
										learning_rate_decay = 0.999, update = 'rmsprop', verbose=True, dropout=1.0)
net.set_level_learning_parameters(1, reg = 0.0000, learning_rate = 0.0015, batch_size = 250, num_epochs = 5, 
										learning_rate_decay = 0.999, update = 'rmsprop', verbose=True, dropout=1.0)
net.set_level_learning_parameters(2, reg = 0.0000, learning_rate = 0.0015, batch_size = 250, num_epochs = 5, 
										learning_rate_decay = 0.999, update = 'rmsprop', verbose=True, dropout=1.0)

print "Finished setting parameters"

if net.check_level_continuity(X):
	print ""
	print """ __  .___  ___.      ___       _______  __  .__   __.  _______  _______          _______..______    _______  _______   ______  __    __  
|  | |   \/   |     /   \     /  _____||  | |  \ |  | |   ____||       \        /       ||   _  \  |   ____||   ____| /      ||  |  |  | 
|  | |  \  /  |    /  ^  \   |  |  __  |  | |   \|  | |  |__   |  .--.  |      |   (----`|  |_)  | |  |__   |  |__   |  ,----'|  |__|  | 
|  | |  |\/|  |   /  /_\  \  |  | |_ | |  | |  . `  | |   __|  |  |  |  |       \   \    |   ___/  |   __|  |   __|  |  |     |   __   | 
|  | |  |  |  |  /  _____  \ |  |__| | |  | |  |\   | |  |____ |  '--'  |   .----)   |   |  |      |  |____ |  |____ |  `----.|  |  |  | 
|__| |__|  |__| /__/     \__\ \______| |__| |__| \__| |_______||_______/    |_______/    | _|      |_______||_______| \______||__|  |__|"""
	print ""
else:
	print "It didn't work"