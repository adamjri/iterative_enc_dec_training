from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

import numpy as np
from keras.models import Sequential
from keras.layers import Input

class DualSigmoid(Activation):
	def __init__(self, activation, **kwargs):
		super(DualSigmoid, self).__init__(activation, **kwargs)
		self.__name__ = 'DualSigmoid'

def dual_sigmoid(x):
	value_plus = K.ones_like(x)*5
	value_minus = K.ones_like(x)*-5
	return (K.sigmoid(x+value_plus)+K.sigmoid(x+value_minus))*0.5

get_custom_objects().update({'DualSigmoid': DualSigmoid(dual_sigmoid)})
