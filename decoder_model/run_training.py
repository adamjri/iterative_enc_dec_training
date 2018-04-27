try:
	from decoder_model.models import *
except:
	from models import *

try:
	from decoder_model.model_params import get_model_params
except:
	from model_params import get_model_params

try:
	from decoder_model.datasets import load_dataset_from_classifier, parse_train_test_datasets
except:
	from datasets import load_dataset_from_classifier, parse_train_test_datasets

from keras.optimizers import Adam, SGD, RMSprop, Nadam

import sys
import argparse
import os

MyModelClass = simple_dense.SimpleDenseModel
Visualizer = visualizers.simple_dense_visualizer

def train_loop(input_dict):
	model_params = input_dict["model_params"]
	model_params["optimizer"] = Adam(**model_params["optimizer_args"])
	train_dataset = input_dict["train_dataset"]
	train_batch_size = input_dict["train_batch_size"]
	test_dataset = input_dict["test_dataset"]
	test_batch_size = input_dict["test_batch_size"]
	num_epochs = input_dict["num_epochs"]
	epochs_per_test = input_dict["epochs_per_test"]
	num_test_vis = input_dict["num_test_vis"]
	output_dir = input_dict["output_dir"]
	prefix = input_dict["prefix"]


	model = MyModelClass(model_params)
	model.load_model()
	model.train(train_dataset, train_batch_size,
			test_dataset, test_batch_size,
			num_epochs, epochs_per_test,
			num_test_vis=num_test_vis, visualizer=Visualizer,
			output_dir=output_dir, prefix=prefix, verbosity=2)

def get_train_params(output_dir, results_dir, depth,
					num_epochs = 5000, train_batch_size = 64, test_batch_size = 64,
					epochs_per_test = 50, prefix = None):
	params_dict = {}
	train_dataset, test_dataset = parse_train_test_datasets(load_dataset_from_classifier(results_dir))
	input_size = len(train_dataset["X"][0])
	output_size = len(train_dataset["Y"][0])
	train_data_size = len(train_dataset["X"])
	test_data_size = len(test_dataset["X"])
	params_dict["model_params"] = get_model_params(input_size, output_size, depth, train_data_size)
	params_dict["train_dataset"] = train_dataset
	params_dict["train_batch_size"] = min(train_batch_size, train_data_size/10)
	params_dict["test_dataset"] = test_dataset
	params_dict["test_batch_size"] = min(test_batch_size, test_data_size/10)
	params_dict["num_epochs"] = num_epochs
	params_dict["epochs_per_test"] = epochs_per_test
	params_dict["num_test_vis"] = 1
	params_dict["output_dir"] = output_dir
	params_dict["prefix"] = prefix
	return params_dict

def get_train_params_with_dataset(output_dir, train_dataset, test_dataset, depth,
					num_epochs = 5000, train_batch_size = 64, test_batch_size = 64,
					epochs_per_test = 50, prefix = None):
	params_dict = {}
	input_size = len(train_dataset["X"][0])
	output_size = len(train_dataset["Y"][0])
	train_data_size = len(train_dataset["X"])
	test_data_size = len(test_dataset["X"])
	params_dict["model_params"] = get_model_params(input_size, output_size, depth, train_data_size)
	params_dict["train_dataset"] = train_dataset
	params_dict["train_batch_size"] = min(train_batch_size, train_data_size/10)
	params_dict["test_dataset"] = test_dataset
	params_dict["test_batch_size"] = min(test_batch_size, test_data_size/10)
	params_dict["num_epochs"] = num_epochs
	params_dict["epochs_per_test"] = epochs_per_test
	params_dict["num_test_vis"] = 1
	params_dict["output_dir"] = output_dir
	params_dict["prefix"] = prefix
	return params_dict

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output_dir')
	parser.add_argument('-i', '--input_dir')
	parser.add_argument('-d', '--depth', type=int, default=1)
	parser.add_argument('-t', '--trials', type=int, default=1)
	return parser.parse_args()

if __name__=="__main__":
	args = parse_args()

	for i in range(args.trials):
		tp = get_train_params(args.output_dir, args.input_dir, args.depth, prefix="trial_"+str(i))
		train_loop(tp)
