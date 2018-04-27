try:
	from classifier_model.models import *
except:
	from models import *

try:
	from classifier_model.model_params import *
except:
	from model_params import *

try:
	from classifier_model.datasets import parse_train_test_datasets
except:
	from datasets import parse_train_test_datasets

try:
	from classifier_model.utils import load_dataset_from_file
except:
	from utils import load_dataset_from_file

from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.models import model_from_json

import sys
import argparse
import os
import random
import numpy as np

MyModelClass = simple_dense.SimpleDenseModel
Visualizer = visualizers.simple_dense_visualizer

def train_loop(input_dict):
	model_params = input_dict["model_params"]
	model_params["optimizer"] = Adam(**model_params["optimizer_args"])
	train_dataset = input_dict["train_dataset"]
	train_batch_size = input_dict["train_batch_size"]
	test_dataset = input_dict["test_dataset"]
	test_batch_size = input_dict["test_batch_size"]
	train_s_dataset = input_dict["train_s_dataset"]
	test_s_dataset = input_dict["test_s_dataset"]
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
			output_dir=output_dir,
			train_s_dataset=train_s_dataset, test_s_dataset=test_s_dataset,
			prefix=prefix, verbosity=2)

def get_train_params(output_dir, dataset_file, depth,
					num_epochs = 5000, train_batch_size = 64, test_batch_size = 64,
					epochs_per_test = 50, prefix = None):
	params_dict = {}
	train_dataset, test_dataset = parse_train_test_datasets(dataset_file, train_ratio=0.02)
	input_size = len(train_dataset["X"][0])
	output_size = len(train_dataset["Y"][0])
	train_data_size = len(train_dataset["X"])
	test_data_size = len(test_dataset["X"])
	params_dict["model_params"] = get_model_params(input_size, output_size, depth, train_data_size)
	params_dict["train_dataset"] = train_dataset
	params_dict["train_batch_size"] = min(train_batch_size, train_data_size/10)
	params_dict["test_dataset"] = test_dataset
	params_dict["test_batch_size"] = min(test_batch_size, test_data_size/10)
	params_dict["train_s_dataset"] = None
	params_dict["test_s_dataset"] = None
	params_dict["num_epochs"] = num_epochs
	params_dict["epochs_per_test"] = epochs_per_test
	params_dict["num_test_vis"] = 1
	params_dict["output_dir"] = output_dir
	params_dict["prefix"] = prefix
	return params_dict

def get_train_params_with_dataset(output_dir, train_dataset, test_dataset, depth,
								train_s_dataset=None, test_s_dataset=None,
								num_epochs = 5000, train_batch_size = 64, test_batch_size = 64,
								epochs_per_test = 50, prefix = None, tri_class=False):
	params_dict = {}
	input_size = len(train_dataset["X"][0])
	output_size = len(train_dataset["Y"][0])
	train_data_size = len(train_dataset["X"])
	if not train_s_dataset is None:
		train_data_size += len(train_s_dataset["X"])
	test_data_size = len(test_dataset["X"])
	if not test_s_dataset is None:
		test_data_size += len(test_s_dataset["X"])
	if tri_class:
		params_dict["model_params"] = get_l2_tri_model_params(input_size, output_size, depth, train_data_size)
	else:
		params_dict["model_params"] = get_model_params(input_size, output_size, depth, train_data_size)
	params_dict["train_dataset"] = train_dataset
	params_dict["train_batch_size"] = min(train_batch_size, train_data_size/10)
	params_dict["test_dataset"] = test_dataset
	params_dict["test_batch_size"] = min(test_batch_size, test_data_size/10)
	params_dict["train_s_dataset"] = train_s_dataset
	params_dict["test_s_dataset"] = test_s_dataset
	params_dict["num_epochs"] = num_epochs
	params_dict["epochs_per_test"] = epochs_per_test
	params_dict["num_test_vis"] = 1
	params_dict["output_dir"] = output_dir
	params_dict["prefix"] = prefix
	return params_dict

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output_dir')
	parser.add_argument('-i', '--input_data')
	parser.add_argument('-d', '--depth', type=int, default=1)
	parser.add_argument('-t', '--trials', type=int, default=1)
	return parser.parse_args()

# ******************************************************************************************
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_best_epoch(test_log_file):
	log_dataset, log_header = load_dataset_from_file(test_log_file)
	min_loss_index = np.argmin(np.array(log_dataset["loss"]))
	return log_dataset["epoch"][min_loss_index]

def load_model_from_results_dir(results_dir, model_json_file = "model.json",
												model_h5_file = "model.h5",
												test_log_file = "test_log.txt"):
	test_log_file = os.path.join(results_dir, test_log_file)
	model_json_file = os.path.join(results_dir, model_json_file)

	best_epoch = get_best_epoch(test_log_file)
	epoch_dir = os.path.join(results_dir, "Epoch_"+str(best_epoch))
	model_h5_file = os.path.join(epoch_dir, model_h5_file)

	# load json
	json_f = open(model_json_file, 'r')
	model_json = json_f.read()
	json_f.close()

	# create model
	model = model_from_json(model_json)

	# load weights
	model.load_weights(model_h5_file)

	return model

def ordinal_tri_classification(output_dir, train_dataset, test_dataset, separator_data, num_s_points, depth):
	s_inputs = random.sample(separator_data["separator"], num_s_points)
	s_labels_0 = [np.array([0]) for j in range(num_s_points)]
	s_labels_1 = [np.array([1]) for j in range(num_s_points)]
	train_s_dataset_0 = {"X": s_inputs, "Y": s_labels_0}
	train_s_dataset_1 = {"X": s_inputs, "Y": s_labels_1}
	tp_0 = get_train_params_with_dataset(output_dir, train_dataset, test_dataset,
											depth, train_s_dataset=train_s_dataset_0,
											prefix = "model_0")
	tp_1 = get_train_params_with_dataset(output_dir, train_dataset, test_dataset,
											depth, train_s_dataset=train_s_dataset_1,
											prefix = "model_1")
	train_loop(tp_0)
	train_loop(tp_1)
	sub_dirs = get_immediate_subdirectories(output_dir)
	for sub_dir in sub_dirs:
		if sub_dir[0:7]=="model_0":
			model_0_dir = os.path.join(output_dir, sub_dir)
		if sub_dir[0:7]=="model_1":
			model_1_dir = os.path.join(output_dir, sub_dir)
	model_0 = load_model_from_results_dir(model_0_dir)
	model_1 = load_model_from_results_dir(model_1_dir)
	inputs = np.array(test_dataset["X"])
	results_0 = model_0.predict(inputs)
	results_1 = model_1.predict(inputs)
	total = len(results_0)
	num_correct = 0
	for i in range(total):
		if results_0[i][0]>=0.5:
			result_0 = 1
		else:
			result_0 = 0
		if results_1[i][0]>=0.5:
			result_1 = 1
		else:
			result_1 = 0
		label = test_dataset["Y"][i][0]
		if result_0 == result_1:
			if result_0 == label:
				num_correct+=1
	accuracy = float(num_correct)/float(total)
	accuracy_file = os.path.join(output_dir, "accuracy.txt")
	f = open(accuracy_file, 'w')
	f.write(str(accuracy))
	f.close()

def tri_class_test():
	args = parse_args()
	separator_data, header = load_dataset_from_file("/scratch/richards/generative_data/test/data_processed_4.txt")
	num_s_points = 400
	for i in range(args.trials):
		trial_output_dir = os.path.join(args.output_dir, "trial_"+str(i))
		tp = get_train_params(trial_output_dir, args.input_data, args.depth, prefix="binary")
		train_loop(tp)
		ordinal_tri_classification(trial_output_dir, tp["train_dataset"], tp["test_dataset"],
									separator_data, num_s_points, args.depth)

# ******************************************************************************************

def l2_tri_classification(output_dir, train_dataset, test_dataset, separator_data, num_s_points, depth):
	s_inputs = random.sample(separator_data["separator"], num_s_points)
	s_labels = [np.array([0.5]) for j in range(num_s_points)]
	train_s_dataset = {"X": s_inputs, "Y": s_labels}
	tp = get_train_params_with_dataset(output_dir, train_dataset, test_dataset,
										depth, train_s_dataset=train_s_dataset,
										prefix="tri_class", tri_class=True)
	# tp = get_train_params_with_dataset(output_dir, train_dataset, test_dataset,
	# 									depth,
	# 									prefix="tri_class", tri_class=True)
	train_loop(tp)

def l2_tri_class_test():
	args = parse_args()
	separator_data, header = load_dataset_from_file("/scratch/richards/generative_data/test3/data_processed_3.txt")
	num_s_points = 400
	for i in range(args.trials):
		trial_output_dir = os.path.join(args.output_dir, "trial_"+str(i))
		tp = get_train_params(trial_output_dir, args.input_data, args.depth, prefix="binary")
		train_loop(tp)
		l2_tri_classification(trial_output_dir, tp["train_dataset"], tp["test_dataset"],
								separator_data, num_s_points, args.depth)
if __name__=="__main__":
	# tri_class_test()
	l2_tri_class_test()
	# *****************************
