class MultiLevelConvNet():

	def __init__(self, X_train, X_val, numLevels, component_dim = (1, 67, 100)):
		self.numLevels = numLevels
		self.X_train = X_train
		self.X_val = X_val
		self.component_dim = component_dim
		self.levels = {{}}
		self.trainer = ClassifierTrainer()

	def set_level_parameters(n, fn, model, y_train, y_val, input_dim, numPreviousComponents, stride):
		self.levels[n]['fn'] = fn
		self.levels[n]['model'] = model
		self.levels[n]['y_train'] = y_train
		self.levels[n]['y_val'] = y_val
		self.levels[n]['input_dim'] = input_dim
		self.levels[n]['stride'] = stride
		if n != 0: self.levels[n]['numPreviousComponents'] = numPreviousComponents

	def set_level_learning_parameters(n, reg = 0.0000, learning_rate = 0.0015, batch_size = 250, num_epochs = 5, 
										learning_rate_decay = 0.999, update = 'rmsprop', verbose=True, dropout=1.0):
		self.levels[n]['reg'] = reg
		self.levels[n]['learning_rate'] = learning_rate
		self.levels[n]['batch_size'] = batch_size
		self.levels[n]['num_epochs'] = num_epochs
		self.levels[n]['learning_rate_decay'] = learning_rate_decay
		self.levels[n]['update'] = update
		self.levels[n]['verbose'] = verbose
		self.levels[n]['dropout'] = dropout

	def check_level_continuity():
		return NotImplementedError()

	def process_to_level(n, X):
		for i in range(n-1)
			

		if n == 0:
			return X
		A = self.levels[n-1]['fn'](process_to_level(n-1), extract_features = True)


	def predict_level(n, X_train=False, X_val = False, X=None):
		if X_train:
			X = X_train
		elif X_val:
			X = X_val
		else:
			X = X

		for i in range(n-1):
			X = forward_level(i, X)

		return self.levels[n]['fn'](X, return_probs = True)

	def train_level(n):
		X_train, X_val = self.X_train, self.X_val

		X_train = process_to_level(n, self.X_train)
		X_val = process_to_level(n, self.X_val)

		y_train, y_val = self.levels[n]['y_train'], self.levels[n]['y_val']
		model, fn = self.levels[n]['model'], self.levels['fn']

		results = self.trainer.train(X_train, y_train, X_val, y_val, model, fn,
          	reg=self.levels[n]['reg'], learning_rate=self.levels[n]['learning_rate'], batch_size=self.levels[n]['batch_size'] num_epochs=self.levels[n]['num_epochs'],
          	learning_rate_decay=self.levels[n]['learning_rate_decay'], update=self.levels[n]['update'], verbose=self.levels[n]['verbose'], dropout=self.levels[n]['dropout'])

		best_model = results[0]
		self.levels[n]['model'] = best_model

	def split(X, num):
		size = X.shape[3] / num
		return np.array([X[:, :, :, i : i + size] for i in range(num)]), size

	def adjoin(features, num):
		return np.reshape(features, (features[0]/num, features[1], features[2], features[3]*num))



