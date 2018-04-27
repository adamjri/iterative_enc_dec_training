import numpy as np
import os
import sys

try:
	from separator_correction.utils import load_dataset_from_file
except:
	from utils import load_dataset_from_file

try:
	from separator_correction.get_separator_hp import get_separator_hp, get_best_epoch
except:
	from get_separator_hp import get_separator_hp, get_best_epoch

try:
	from separator_correction.separator_points import *
except:
	from separator_points import *

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_corrected_dataset_for_decoder(results_dir, model_json_file = "model.json",
												model_h5_file = "model.h5",
												test_log_file = "test_log.txt",
												layer_data_file = "outputs_to_labels.txt"):

	# get separating hyperplane [N, R]
	HP = get_separator_hp(results_dir, model_json_file = model_json_file,
										model_h5_file = model_h5_file,
										test_log_file = test_log_file)

	# get dataset from final layer
	test_log_file = os.path.join(results_dir, test_log_file)
	best_epoch = get_best_epoch(test_log_file)
	epoch_dir = os.path.join(results_dir, "Epoch_"+str(best_epoch))
	layer_dirs = get_immediate_subdirectories(epoch_dir)
	final_layer_dir = os.path.join(epoch_dir, "layer_"+str(len(layer_dirs)-1))
	first_layer_dir = os.path.join(epoch_dir, "layer_0")
	final_layer_data_file = os.path.join(final_layer_dir, layer_data_file)
	first_layer_data_file = os.path.join(first_layer_dir, layer_data_file)
	final_layer_dataset, final_layer_header = load_dataset_from_file(final_layer_data_file)
	first_layer_dataset, first_layer_header = load_dataset_from_file(first_layer_data_file)

	# modify data using hyperplane
	train_points = final_layer_dataset["train_X"]
	train_outputs = final_layer_dataset["train_Y"]
	train_labels = final_layer_dataset["train_GT"]
	test_points = final_layer_dataset["test_X"]
	test_outputs = final_layer_dataset["test_Y"]
	test_labels = final_layer_dataset["test_GT"]

	train_correction_list = get_correction_list(train_outputs, train_labels)
	test_correction_list = get_correction_list(test_outputs, test_labels)

	train_full_s_points = get_s_points(train_points, HP)
	test_full_s_points = get_s_points(test_points, HP)

	# s points
	train_s_points = get_s_point_set(train_full_s_points, train_correction_list)
	test_s_points = get_s_point_set(test_full_s_points, test_correction_list)

	# y_c
	train_corrected_points = get_corrected_points(train_points, train_full_s_points, train_correction_list)
	test_corrected_points = get_corrected_points(test_points, test_full_s_points, test_correction_list)

	is_single_node = len(train_outputs[0])==1
	s_label = get_s_label(is_single_node)

	# s labels
	train_s_labels = [s_label for i in range(len(train_s_points))]
	test_s_labels = [s_label for i in range(len(test_s_points))]

	# get corresponding input data
	train_inputs = first_layer_dataset["train_X"]
	test_inputs = first_layer_dataset["test_X"]

	train_corrected_dataset = {}
	train_corrected_dataset["X"] = train_corrected_points
	train_corrected_dataset["Y"] = train_inputs
	train_corrected_dataset["s_points"] = train_s_points
	train_corrected_dataset["s_labels"] = train_s_labels

	test_corrected_dataset = {}
	test_corrected_dataset["X"] = test_corrected_points
	test_corrected_dataset["Y"] = test_inputs
	test_corrected_dataset["s_points"] = test_s_points
	test_corrected_dataset["s_labels"] = test_s_labels

	return [train_corrected_dataset, test_corrected_dataset]

if __name__ == "__main__":
	results_dir = "/scratch/richards/generative_data/full_test_4/classifier_0_20180307141035"
	get_corrected_dataset_for_decoder(results_dir)
