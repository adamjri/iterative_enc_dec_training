''' Using weight initialization procedures from:
https://arxiv.org/abs/1704.08863 '''

from keras.initializers import *

def kumar_normal(activation, seed=None):
	if activation=="tanh":
		scale = 1.0
	elif activation=="sigmoid":
		scale = 3.6
	elif activation=='relu':
		scale = 2.0
	elif activation=='softplus':
		scale = 2.0
	else:
		scale = 1.0
	return VarianceScaling(scale=scale,
						mode='fan_avg',
						distribution='normal',
						seed=seed)

def kumar_uniform(activation, seed=None):
	if activation=="tanh":
		scale = 1.0
	elif activation=="sigmoid":
		scale = 13.0
	elif activation=='relu':
		scale = 2.0
	elif activation=='softplus':
		scale = 2.0
	else:
		scale = 1.0
	return VarianceScaling(scale=scale,
							mode='fan_avg',
							distribution='uniform',
							seed=seed)

def my_bias_initializer(activation, seed=None):
	if activation=="relu":
		value = 2.0
	elif activation=="softplus":
		value = 0.5
	else:
		value = 0.0
	return Constant(value=value)
