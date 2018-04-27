try:
	from generative_model.models.simple_dense import SimpleDenseModel
except:
	from models.simple_dense import SimpleDenseModel

try:
	from generative_model.utils import load_dataset_from_file, write_dataset_to_file
except:
	from utils import load_dataset_from_file, write_dataset_to_file

try:
	from generative_model.plotting.plot_dataset import plot_2d_dataset_colored
except:
	from plotting.plot_dataset import plot_2d_dataset_colored

from keras.initializers import *
import os
import numpy as np
import sys


if __name__ == "__main__":
	depth = int(sys.argv[1])
	num_generators = int(sys.argv[2])
	input_dataset_dir = "/scratch/richards/generative_data/classifier_datasets/"
	output_dataset_dir = "/scratch/richards/generative_data/generator_datasets/"
	filename_input = os.path.join(input_dataset_dir, "1d_data_input.txt")
	output_dir = os.path.join(output_dataset_dir, "depth_"+str(depth))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	params = {
		"layers": [2 for i in range(depth+1)],
		"activations": ["sigmoid" for i in range(depth)],
		"has_biases": [True for i in range(depth)],
		"initializers": [[RandomNormal(stddev=1.0),
						  RandomNormal(stddev=1.0)]
						  for i in range(depth)]
	}
	model = SimpleDenseModel(params)
	dataset, header = load_dataset_from_file(filename_input)
	generator_dataset = {"X":[], "Y":[]}

	for i in range(num_generators):
		model.load_model()
		outfile = os.path.join(output_dir, "dataset_"+str(i)+".txt")

		new_dataset = {}
		new_dataset["X"] = model.get_outputs(dataset["X"])
		new_dataset["Y"] = dataset["Y"]

		generator_dataset["X"]+=new_dataset["X"]
		label = np.array([0 for j in range(num_generators)])
		label[i] = 1
		generator_dataset["Y"]+=[label for j in range(len(new_dataset["X"]))]

		write_dataset_to_file(outfile, new_dataset)

	generator_data_file = os.path.join(output_dataset_dir, "generator_dataset_"+str(depth)+".txt")
	write_dataset_to_file(generator_data_file, generator_dataset)
