class MultiLevelConvNet():

	def __init__(self, numLevels):
		self.numLevels = numLevels
		self.levels = {{}}
		self.trainer = ClassifierTrainer()

	def set_level_parameters(self, n, fn, model, y_train, y_val, component_dim, numComponents, stride):
		self.levels[n]['fn'] = fn
		self.levels[n]['model'] = model
		self.levels[n]['component_dim'] = component_dim
		self.levels[n]['stride'] = stride
		self.levels[n]['numComponents'] = numComponents

	def set_level_learning_parameters(self, n, reg = 0.0000, learning_rate = 0.0015, batch_size = 250, num_epochs = 5, 
										learning_rate_decay = 0.999, update = 'rmsprop', verbose=True, dropout=1.0):
		self.levels[n]['reg'] = reg
		self.levels[n]['learning_rate'] = learning_rate
		self.levels[n]['batch_size'] = batch_size
		self.levels[n]['num_epochs'] = num_epochs
		self.levels[n]['learning_rate_decay'] = learning_rate_decay
		self.levels[n]['update'] = update
		self.levels[n]['verbose'] = verbose
		self.levels[n]['dropout'] = dropout

	def check_level_continuity(self, X):
		self.predict_level(self.numLevels-1, X)

		return True

	def forward_level(self, n, X):
		component_dims = self.levels[n]['component_dim']
		stride = self.levels[n]['stride']
		numComponents = self.levels[n]['numComponents']
		current = 0
		nextLevel = []

		while (current + component_dims[2] + stride*(numComponents-1) < X.shape[3]):
			components = []
			for i in range(numComponents):
				component = X[:, :component_dims[0], :component_dims[1], current : current + component_dims[2]]
				components.append(component)
				current += stride
			A = self.levels[n]['fn'](reduce(lambda x, y: np.hstack(x, y), components), extract_features = True)
			nextLevel.append(A)

		return reduce(lambda x, y: np.hstack(x, y), components)


	def process_to_level(self, n, X):
		for i in range(n-1):
			X = self.forward_level(i, X)
		return X


	def predict_level(self, n, X):
		A = self.process_to_level(n, X)

		return self.levels[n]['fn'](A, return_probs = True)

	def train_level(self, n, X_train, X_val, y_train, y_val):
		X_train = self.process_to_level(n, self.X_train)
		X_val = self.process_to_level(n, self.X_val)

		model, fn = self.levels[n]['model'], self.levels['fn']

		dims = self.levels[n]['component_dim']

		X_train = reduce(lambda x, y: np.vstack(x, y), [X_train[:, :dims[0], :dims[1], i : i + dims[2]] for i in range(X_train.shape[3] / dims[2])])
		X_val = reduce(lambda x, y: np.vstack(x, y), [X_val[:, :dims[0], :dims[1], i : i + dims[2]] for i in range(X_val.shape[3] / dims[2])])
		y_train = np.reshape(y_train, (y_train.size))
		y_val = np.reshape(y_val, (y_val.size))

		results = self.trainer.train(X_train, y_train, X_val, y_val, model, fn,
          	reg=self.levels[n]['reg'], learning_rate=self.levels[n]['learning_rate'], batch_size=self.levels[n]['batch_size'], num_epochs=self.levels[n]['num_epochs'],
          	learning_rate_decay=self.levels[n]['learning_rate_decay'], update=self.levels[n]['update'], verbose=self.levels[n]['verbose'], dropout=self.levels[n]['dropout'])

		best_model = results[0]
		self.levels[n]['model'] = best_model



