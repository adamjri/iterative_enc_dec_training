try:
	from generative_model.data.utils import assert_key_in_dict
except:
	try:
		from data.utils import assert_key_in_dict
	except:
		from utils import assert_key_in_dict

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

import sys

class SamplerBase:
	__metaclass__ = ABCMeta

	def __init__(self, params_dict):
		np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
		self.params = params_dict
		self.verify_params
		assert_key_in_dict("s_dimension", params_dict)
		assert_key_in_dict("dimension", params_dict)
		assert_key_in_dict("bounds", params_dict)
		self.s_dimension = self.params["s_dimension"]
		self.dimension = self.params["dimension"]
		self.bounds = self.params["bounds"]
		if not len(self.bounds)==self.dimension:
			print "Bounds must have the proper dimension"
			print "Bounds: "+str(self.bounds)
			print "Dimension: "+str(self.dimension)
			sys.exit()
		for i in range(len(self.bounds)):
			if len(self.bounds[i])!=2:
				print "Each bound must have exactly a min and max"
				sys.exit()
			min_ = min(self.bounds[i])
			max_ = max(self.bounds[i])
			self.bounds[i] = [min_, max_]

		if not "homogenous_transform" in self.params:
			self.homogenous_transform = np.eye(self.s_dimension+1)
		else:
			self.homogenous_transform = self.params["homogenous_transform"]
			# check transform for validity
			if not self.homogenous_transform.shape==(self.s_dimension+1, self.s_dimension+1):
				print "homogenous_transform has invalid shape: "\
						+str(self.homogenous_transform.shape) + " should be "\
						+str((self.s_dimension+1, self.s_dimension+1))
				sys.exit()
			else:
				proper_bottom_row = np.array([0.0 for i in range(self.s_dimension)]+[1.0])
				if not np.array_equiv(self.homogenous_transform[-1,:],proper_bottom_row):
					print "homogenous_transform must have properly formatted bottom row: "\
							+str(self.homogenous_transform[-1]) + " should be "\
							+str(np.array([0.0 for i in range(self.s_dimension)]+[1.0]))
					sys.exit()
			if np.linalg.det(self.homogenous_transform[:-1, :-1])**2 < 0.00000001:
				print "homogenous_transform must be invertible"
				sys.exit()



	@abstractmethod
	def verify_params(self):
		pass

	@abstractmethod
	def get_sample(self):
		pass

	def generate_samples(self, num_samples):
		return [self.get_sample() for i in range(num_samples)]
