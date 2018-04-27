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

class OneDSampler(SamplerBase):
	def verify_params(self):
		if not self.params["dimension"]==1:
			print("Dimension must be 1")

	def get_sample(self):
		point = np.array([random.uniform(self.bounds[0][0], self.bounds[0][1])]
						+[0.0 for i in range(self.params['s_dimension']-1)])
		return transform(self.homogenous_transform, point)



if __name__ == "__main__":
	params_dict = {
		"dimension": 1,
		"s_dimension": 2,
		"bounds": [[0,1]]
	}

	sampler = OneDSampler(params_dict)
	print sampler.generate_samples(20)
