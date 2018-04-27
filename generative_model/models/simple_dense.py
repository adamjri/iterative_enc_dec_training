try:
	from generative_model.models.utils import assert_key_in_dict
except:
	try:
		from models.utils import assert_key_in_dict
	except:
		from utils import assert_key_in_dict

import numpy as np
import math
import random
import sys

# activation is 4/pi * arctan
class SimpleDenseModel():
	def __init__(self, params):
		self.model=None
		self.weights = None
		self.biases = None
		self.params = params
		# verify params
		self.verify_params()
		self.activations=[np.vectorize(f, otypes=[np.float]) for f in params["activations"]]
		self.load_model()

	def verify_params(self):
		assert_key_in_dict("layers", self.params)
		assert_key_in_dict("activations", self.params)
		if not len(self.params["layers"])-1 == len(self.params["activations"]):
			print "Num activations must equal num layers-1"
			sys.exit()

	def load_model(self):
		print "Loading Model..."
		self.model = None
		self.weights = None
		self.biases = None
		self.weights = []
		self.biases = []
		for i in range(len(self.params["layers"])-1):
			row_len = self.params["layers"][i]
			col_len = self.params["layers"][i+1]
			weight = np.array([[random.gauss(0.0, 1.0) for j in range(row_len)]
							   for k in range(col_len)])
			bias = np.array([random.gauss(0.0, 0.0) for j in range(col_len)])
			w_scale = abs(np.linalg.det(weight))
			self.weights.append(weight/w_scale)
			self.biases.append(bias)
		def g(x):
			v = x
			for i, weight in enumerate(self.weights):
				v = np.dot(weight, v)
				v += self.biases[i]
				v = self.activations[i](v)
			return v
		self.model = g

	def get_outputs(self, inputs):
		outputs = []
		for input_ in inputs:
			outputs.append(self.model(input_))
		return outputs
