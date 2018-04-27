try:
	from classifier_model.utils import load_dataset_from_file
except:
	from utils import load_dataset_from_file

import random
import numpy as np
import os
import sys

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_best_epoch(test_log_file):
	log_dataset, log_header = load_dataset_from_file(test_log_file)
	min_loss_index = np.argmin(np.array(log_dataset["loss"]))
	return log_dataset["epoch"][min_loss_index]

def load_dataset_from_classifier(classifier_result_dir, test_log_file="test_log.txt",
														test_data_file="test_data.txt",
														layer_data_file="outputs_to_labels.txt",
														):
	test_log_file = os.path.join(classifier_result_dir, test_log_file)
	best_epoch = get_best_epoch(test_log_file)
	epoch_dir = os.path.join(classifier_result_dir, "Epoch_"+str(best_epoch))
	layer_dirs = get_immediate_subdirectories(epoch_dir)
	input_layer_dir = os.path.join(epoch_dir, "layer_0")
	final_layer_dir = os.path.join(epoch_dir, "layer_"+str(len(layer_dirs)-1))

	input_layer_data_file = os.path.join(input_layer_dir, layer_data_file)
	final_layer_data_file = os.path.join(final_layer_dir, layer_data_file)

	input_layer_dataset, input_layer_header = load_dataset_from_file(input_layer_data_file)
	final_layer_dataset, final_layer_header = load_dataset_from_file(final_layer_data_file)

	dataset = {}
	dataset["X"] = final_layer_dataset["train_X"]+final_layer_dataset["test_X"]
	dataset["Y"] = input_layer_dataset["train_X"]+input_layer_dataset["test_X"]

	# get separator if possible
	if "separator_X" in final_layer_dataset:
		dataset["separator"] = final_layer_dataset["separator_X"]
	test_dataset, test_header = load_dataset_from_file(os.path.join(classifier_result_dir, test_data_file))
	if "separator" in test_dataset:
		dataset["separator_GT"] = test_dataset["separator"]

	return dataset

def parse_train_test_datasets(dataset, train_ratio=0.85):
	num_samples = len(dataset["X"])
	num_train = int(train_ratio*num_samples)
	population = list(range(num_samples))
	train_samples = random.sample(population, num_train)
	train_samples_sorted = np.sort(train_samples)
	test_samples = []
	i = 0
	pointer = 0
	while True:
		if i==num_samples:
			break
		if pointer==len(train_samples_sorted):
			test_samples.append(i)
			i+=1
			continue
		if train_samples_sorted[pointer]<i:
			pointer+=1
		elif train_samples_sorted[pointer]==i:
			pointer+=1
			i+=1
		else:
			test_samples.append(i)
			i+=1

	random.shuffle(test_samples)

	train_dataset = {"X":[], "Y":[]}
	for sample in train_samples:
		train_dataset["X"].append(dataset["X"][sample])
		train_dataset["Y"].append(dataset["Y"][sample])
	test_dataset = {"X":[], "Y":[]}
	for sample in test_samples:
		test_dataset["X"].append(dataset["X"][sample])
		test_dataset["Y"].append(dataset["Y"][sample])

	for key in dataset:
		if key!="X" and key!="Y":
			train_dataset[key] = dataset[key]
			test_dataset[key] = dataset[key]
	return [train_dataset, test_dataset]
