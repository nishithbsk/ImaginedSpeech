import

fn1 = speech_convnet
y_train1 = 
y_val1 = 
input_component_dim1 = (1, 64, 100)
output_component_dim1 = (32, )
stride1 = input_component_dim1[2]
num_components1 = 1
model1 = init_speech_convnet(input_shape = (input_component_dim1[0], input_components_dim1[1], input_components_dim1[2] * num_components1), 
							num_classes = 10, filter_size = 3, num_filters = (32, 128, 256), weight_scale = 1e-4)

fn2 = speech_convnet
y_train2 = 
y_val2 = 
input_component_dim2 = output_component_dim1
output_component_dim2 = 
stride2 = input_component_dim2[2]
num_components2 = 5
model2 = init_speech_convnet(input_shape = (input_component_dim2[0], input_components_dim2[1], input_components_dim2[2] * num_components2), 
							num_classes = 2, filter_size = 3, num_filters = (32, 128, 256), weight_scale = 1e-4)

fn3 = speech_convnet
y_train3 = 
y_val3 = 
input_component_dim3 = output_component_dim2
output_component_dim3 = 
stride3 = input_component_dim3[2]
num_components3 = 2
model3 = init_speech_convnet(input_shape = (input_component_dim3[0], input_components_dim3[1], input_components_dim3[2] * num_components3), 
							num_classes = 2, filter_size = 3, num_filters = (32, 128, 256), weight_scale = 1e-4)

##############################################################################################################################################

net = MultiLevelConvNet(X_train, X_val, 3)
net.set_level_parameters(0, fn1, model1, y_train1, y_val1, input_component_dim1, num_components1, stride1)
net.set_level_parameters(1, fn2, model2, y_train2, y_val2, input_component_dim2, num_components2, stride2)
net.set_level_parameters(2, fn3, model3, y_train3, y_val3, input_component_dim3, num_components3, stride3)
net.set_level_learning_parameters(0, reg = 0.0000, learning_rate = 0.0015, batch_size = 250, num_epochs = 5, 
										learning_rate_decay = 0.999, update = 'rmsprop', verbose=True, dropout=1.0)
net.set_level_learning_parameters(1, reg = 0.0000, learning_rate = 0.0015, batch_size = 250, num_epochs = 5, 
										learning_rate_decay = 0.999, update = 'rmsprop', verbose=True, dropout=1.0)
net.set_level_learning_parameters(2, reg = 0.0000, learning_rate = 0.0015, batch_size = 250, num_epochs = 5, 
										learning_rate_decay = 0.999, update = 'rmsprop', verbose=True, dropout=1.0)

if net.check_level_continuity():
	return "IT WORKED"

else:
	return "It didn't work"