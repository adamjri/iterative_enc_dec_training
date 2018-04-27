try:
	from generative_model.utils import write_dataset_to_file, load_dataset_from_file
except:
	from utils import write_dataset_to_file, load_dataset_from_file

try:
	from generative_model.data import *
except:
	from data import *

import numpy as np

def generate_Nd_dataset(params, num_samples, filename=None):
	sampler = simple_Nd_sampler.NDSampler(params)
	labeler = hyperplane_labeling.HPLabeler(sampler)
	midpoint0 = (sampler.bounds[0][0]+sampler.bounds[0][1])/2.0
	midpoint1 = (sampler.bounds[1][0]+sampler.bounds[1][1])/2.0
	HP = labeler.generate_random_HP(center_point=np.array([midpoint0, midpoint1, 0.0]),
									normal_vector=np.array([1.0, 0.0, 0.0]))

	dataset = labeler.generate_dataset(num_samples, HP, separator=True)
	if not filename is None:
		write_dataset_to_file(filename, dataset)
	return dataset

if __name__ == "__main__":
	# filename = "/scratch/richards/generative_data/classifier_datasets/3D/1d_data_input.txt"
	filename = "/scratch/richards/generative_data/test3/input_data.txt"

	params = {
		"dimension": 3,
		"s_dimension": 3,
		"bounds": [[-1, 1], [-1, 1],[-1, 1]]
	}

	num_samples = 5000

	generate_Nd_dataset(params, num_samples, filename=filename)
