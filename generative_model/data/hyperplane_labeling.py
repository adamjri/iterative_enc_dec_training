try:
	from generative_model.data.utils import transform, rotate
except:
	try:
		from data.utils import transform, rotate
	except:
		from utils import transform, rotate

import numpy as np
import scipy.stats as stats
import random
import math
import sys

# scale gives number of std deviations from center to end point
def truncated_centered_gaussian(low, high, scale=2.0):
	mu = (low+high)/2.0
	d = (high-mu)
	return stats.truncnorm(-scale, scale, loc=mu, scale=d/float(scale))

def increment_counter(counter, base):
	counter[0]+=1
	for i in range(len(counter)-1):
		if counter[i]==base:
			counter[i]=0
			counter[i+1]+=1


class HPLabeler:
	def __init__(self, sampler):
		self.sampler = sampler

	def generate_random_HP(self, center_point=None, normal_vector=None, distribution=None):
		concat_arr = np.array([0.0 for i in range(self.sampler.s_dimension-self.sampler.dimension)])
		if distribution is None:
			distribution = truncated_centered_gaussian
		if center_point is None:
			# sample using truncated normal from inside the bounds
			center_point = np.array([distribution(min(bound), max(bound))
									for bound in self.sampler.bounds])
			center_point = np.concatenate((center_point, concat_arr))
			center_point = transform(self.sampler.homogenous_transform, center_point)

		if normal_vector is None:
			normal_vector = np.array([random.uniform(-1.0,1.0)
										for i in range(self.sampler.dimension)])
			normal_vector = np.concatenate((normal_vector, concat_arr))
			normal_vector = rotate(self.sampler.homogenous_transform, normal_vector)

		l2 = np.linalg.norm(normal_vector)
		normal_vector/=l2

		return [center_point, normal_vector]

	def binary_label_from_HP(self, samples, HP):
		labels = []
		for sample in samples:
			v = sample - HP[0]
			d = v.dot(HP[1])
			if d>=0:
				labels.append(np.array([1]))
			else:
				labels.append(np.array([0]))
		return labels

	def generate_separator(self, HP, scale=2, density=20):
		# get rotation matrix to normal vector
		dim = len(HP[0])
		if dim<2:
			print "Space must have dimension>1 to build a separator"
			sys.exit()
		elif dim>8:
			print "Space is too large with dimension>8"
			sys.exit()

		u = np.array([1.0]+[0.0 for i in range(dim-1)])
		y = HP[1]
		v = y - u.dot(y)*u
		if np.linalg.norm(v-np.array([0.0 for i in range(dim)]))<0.000000000000000001:
			R = np.eye(dim)
		else:
			vl2 = np.linalg.norm(v)
			v/=vl2
			cost = u.dot(y)
			sint = math.sqrt(1-cost**2)

			ua = u[:, np.newaxis]
			va = v[:, np.newaxis]
			uva = np.zeros((dim, 2))
			uva[:, 0:1] = ua
			uva[:, 1:2] = va
			RT = np.array([[cost, -sint],[sint, cost]])
			R = np.eye(dim) - ua.dot(ua.T) - va.dot(va.T) + uva.dot(RT).dot(uva.T)

		linspace = np.linspace(-scale, scale, density)
		counter = [0 for i in range(dim-1)]
		points = []
		while counter[-1]<density:
			point = np.array([0.0] + [linspace[counter[i]] for i in range(dim-1)])
			points.append(HP[0]+R.dot(point))
			increment_counter(counter, density)
		return points

	def generate_dataset(self, num_samples, HP, separator=False, scale=2, density=20):
		dataset={}
		dataset["X"] = self.sampler.generate_samples(num_samples)
		dataset["Y"] = self.binary_label_from_HP(dataset["X"], HP)
		if separator:
			dataset["separator"] = self.generate_separator(HP, scale=scale, density=density)
		return dataset

if __name__ == "__main__":
	from simple_Nd_sampler import NDSampler

	params_dict = {
		"dimension": 2,
		"s_dimension": 3,
		"bounds": [[-1,1], [-1,1]]
	}

	num_samples = 20

	sampler = NDSampler(params_dict)
	labeler = HPLabeler(sampler)
	HP = labeler.generate_random_HP(center_point=np.array([0.0,0.0,0.0]),
									normal_vector=np.array([1.0,1.0,0.0]))
	dataset = labeler.generate_dataset(num_samples, HP, separator=True)
	for i in range(num_samples):
		print str(dataset["X"][i])+" -> "+str(dataset["Y"][i])
