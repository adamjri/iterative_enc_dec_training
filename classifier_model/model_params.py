from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.initializers import *

try:
	from classifier_model.models.initializers import *
except:
	from models.initializers import *

import math

import sys

def get_model_params(input_size, output_size, depth, train_data_size):# create model
	'''
	learning rate based on the following empirically discovered values:
	train_data_size -> empirical good learning rate -> computed learning rate
	500 -> 0.0008 -> .000806
	1000 -> 0.0009 -> 0.000926
	2500 -> 0.001 -> 0.00111
	5000 -> 0.0015 -> 0.00128
	'''
	optimizer_args = {'lr': float(train_data_size)**(1.0/5.0)/(4300.0)}

	loss = 'binary_crossentropy'

	metrics = []
	layers = [input_size for i in range(depth)]+[output_size]
	initializers = [[glorot_uniform(), Ones()] for i in range(depth)]
	activations = ['relu' for i in range(depth-1)]+['sigmoid']
	has_biases = [True for i in range(depth)]
	model_params = {"layers": layers,
					"activations": activations,
					"initializers": initializers,
					"has_biases": has_biases,
					"optimizer_args": optimizer_args,
					"loss": loss,
					"metrics": metrics}
	return model_params

def get_l2_tri_model_params(input_size, output_size, depth, train_data_size):# create model
	'''
	learning rate based on the following empirically discovered values:
	train_data_size -> empirical good learning rate -> computed learning rate
	500 -> 0.0008 -> .000806
	1000 -> 0.0009 -> 0.000926
	2500 -> 0.001 -> 0.00111
	5000 -> 0.0015 -> 0.00128
	'''
	optimizer_args = {'lr': float(train_data_size)**(1.0/5.0)/(4300.0)}

	loss = 'mean_squared_error'

	metrics = []
	layers = [input_size for i in range(depth)]+[output_size]
	initializers = [[glorot_uniform(), Ones()] for i in range(depth)]
	activations = ['relu' for i in range(depth-1)]+['DualSigmoid']
	has_biases = [True for i in range(depth)]
	model_params = {"layers": layers,
					"activations": activations,
					"initializers": initializers,
					"has_biases": has_biases,
					"optimizer_args": optimizer_args,
					"loss": loss,
					"metrics": metrics}
	return model_params
