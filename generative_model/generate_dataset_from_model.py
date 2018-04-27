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
	from plotting.plot_dataset import plot_2d_dataset_colored, plot_3d_dataset_colored

import os
import numpy as np
import math
import sys

def relu(x):
	return max(0, x)

def linear(x):
	return x

def atan(x):
	return 2.0*math.atan(x)/3.0

if __name__ == "__main__":
	depth = int(sys.argv[1])
	dataset_dir = "/scratch/richards/generative_data/test3/"
	#dataset_dir = "/scratch/richards/generative_data/classifier_datasets/3D/"
	filename_input = os.path.join(dataset_dir, "input_data.txt")
	input_savefile = os.path.join(dataset_dir, "input_data.png")
	filename_output = os.path.join(dataset_dir, "data_processed_"+str(depth)+".txt")
	savefile = os.path.join(dataset_dir, "data_processed_"+str(depth)+".png")

	dataset, header = load_dataset_from_file(filename_input)
	dimension = len(dataset["X"][0])

	params = {
		"layers": [dimension for i in range(depth+1)],
		"activations": [atan for i in range(depth)]
	}
	model = SimpleDenseModel(params)

	separator=dataset["separator"]

	if not os.path.exists(input_savefile):
		# plot_2d_dataset_colored(dataset, separator=separator, savefile=input_savefile)
		plot_3d_dataset_colored(dataset, separator=separator, savefile=input_savefile)


	new_dataset = {}
	new_dataset["X"] = model.get_outputs(dataset["X"])
	new_dataset["Y"] = dataset["Y"]
	new_dataset["separator"] = model.get_outputs(separator)

	new_separator = new_dataset["separator"]
	write_dataset_to_file(filename_output, new_dataset)
	# plot_2d_dataset_colored(new_dataset, separator=new_separator, savefile=savefile)
	plot_3d_dataset_colored(new_dataset, separator=new_separator, savefile=savefile)
