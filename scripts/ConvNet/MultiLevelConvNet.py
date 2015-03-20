class MultiLevelConvNet():

	def __init__(self, X_train, X_val, numLevels):
		self.numLevels = numLevels
		self.X_train = X_train
		self.X_val = X_val
		self.levels = {{}}
		self.trainer = ClassifierTrainer()

	def set_level_parameters(self, n, fn, model, y_train, y_val, component_dim, numComponents, stride):
		self.levels[n]['fn'] = fn
		self.levels[n]['model'] = model
		self.levels[n]['y_train'] = y_train
		self.levels[n]['y_val'] = y_val
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

	def check_level_continuity():
		self.predict_level(self.numLevels-1, X_train = self.X_train)

		return True

	def forward_level(self, n, X):
		component_dims = self.levels[n]['component_dim']
		stride = self.levels[n]['stride']
		numComponents = self.levels[n]['numComponents']
		current = 0
		nextLevel = []

		while (current + component_dims[2] + stride*(numComponents-1)):
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
			A = self.forward_level(i, X)
		return A


	def predict_level(self, n, X_train=False, X_val = False, X=None):
		if X_train:
			X = X_train
		elif X_val:
			X = X_val
		else:
			X = X

		X = self.process_to_level(n, X)

		return self.levels[n]['fn'](X, return_probs = True)

	def train_level(self, n):
		X_train, X_val = self.X_train, self.X_val

		X_train = self.process_to_level(n, self.X_train)
		X_val = self.process_to_level(n, self.X_val)

		y_train, y_val = self.levels[n]['y_train'], self.levels[n]['y_val']
		model, fn = self.levels[n]['model'], self.levels['fn']

		results = self.trainer.train(X_train, y_train, X_val, y_val, model, fn,
          	reg=self.levels[n]['reg'], learning_rate=self.levels[n]['learning_rate'], batch_size=self.levels[n]['batch_size'] num_epochs=self.levels[n]['num_epochs'],
          	learning_rate_decay=self.levels[n]['learning_rate_decay'], update=self.levels[n]['update'], verbose=self.levels[n]['verbose'], dropout=self.levels[n]['dropout'])

		best_model = results[0]
		self.levels[n]['model'] = best_model



