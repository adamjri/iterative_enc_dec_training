''' Using weight initialization procedures from:
https://arxiv.org/abs/1704.08863 '''

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.utils import plot_model
from keras.initializers import *
from keras import regularizers

import sys

try:
	from classifier_model.models.model_base import ModelBase
except:
	try:
		from models.model_base import ModelBase
	except:
		from model_base import ModelBase

try:
	from classifier_model.models.initializers import *
except:
	try:
		from models.initializers import *
	except:
		from initializers import *

try:
	import classifier_model.models.custom_ops as cust
except:
	try:
		import models.custom_ops as cust
	except:
		import custom_ops as cust

class SimpleDenseModel(ModelBase):
	def verify_params(self):
		necessary_params = ["layers", "initializers", "activations", "has_biases", "optimizer", "loss", "metrics"]
		has_np = True
		for parms in necessary_params:
			if parms not in self.params:
				has_np = False
				print("Missing parameter: "+parms)
		if not has_np:
			sys.exit()

		if len(self.params["layers"])<2:
			print("Must have at least 2 layers")
			sys.exit()
		if len(self.params["layers"]) != len(self.params["activations"])+1:
			print("layers.length must = activations.length+1")
			sys.exit()
		if len(self.params["has_biases"]) != len(self.params["activations"]):
			print("has_biases.length must = activations.length")
			sys.exit()

	def load_model(self, save_file=None):
		print("Loading Model...")
		self.model = None
		input_layer = Input(shape=(self.params["layers"][0],))
		x = input_layer
		# Add hidden layers
		for i in range(1, len(self.params["layers"])):
			x = Dense(self.params["layers"][i], use_bias=self.params["has_biases"][i-1],
						kernel_initializer=self.params["initializers"][i-1][0],
        				bias_initializer=self.params["initializers"][i-1][1],
						activation = self.params["activations"][i-1])(x)
		output_layer = x
		self.model = Model(inputs=input_layer, outputs=output_layer)
		# Compile model
		self.model.compile(optimizer=self.params["optimizer"],
						   loss=self.params["loss"],
						   metrics=self.params["metrics"])
