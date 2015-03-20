import

net = MultiLevelConvNet(X_train, X_val, 3)

model1 = init_three_layer_convnet()
model2 = init_three_layer_convnet()
model3 = init_three_layer_convnet()

net.set_level_parameters()

