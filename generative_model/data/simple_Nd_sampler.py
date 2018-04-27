try:
	from generative_model.data.sampler_base import SamplerBase
except:
	try:
		from data.sampler_base import SamplerBase
	except:
		from sampler_base import SamplerBase

try:
	from generative_model.data.utils import transform
except:
	try:
		from data.utils import transform
	except:
		from utils import transform


import numpy as np
import random

class NDSampler(SamplerBase):
	def verify_params(self):
		if not self.params["dimension"]<=self.params["s_dimension"]:
			print "Sample dimension must be <= space dimension"
			sys.exit()

	def get_sample(self):
		point_in_d = [random.uniform(self.bounds[i][0], self.bounds[i][1])
					for i in range(self.params["dimension"])]
		padding = [0.0 for i in range(self.params['s_dimension']-self.params["dimension"])]
		point = np.array(point_in_d + padding)
		return transform(self.homogenous_transform, point)



if __name__ == "__main__":
	params_dict = {
		"dimension": 2,
		"s_dimension": 3,
		"bounds": [[0,1],[0,1]]
	}

	sampler = NDSampler(params_dict)
	print sampler.generate_samples(20)
