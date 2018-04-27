import os
import sys
import argparse

from utils import load_dataset_from_file, write_dataset_to_file

from classifier_model.datasets import parse_train_test_datasets as pttd

from classifier_model.run_training import get_train_params as ctp
from classifier_model.run_training import get_train_params_with_dataset as ctpd
from classifier_model.run_training import train_loop as ctl

from separator_correction.correct_datasets import get_corrected_dataset_for_decoder as get_cd

from decoder_model.run_training import get_train_params as dtp
from decoder_model.run_training import get_train_params_with_dataset as dtpd
from decoder_model.run_training import train_loop as dtl

from decoder_model.run_inference import load_model_from_results_dir as dlm
from decoder_model.run_inference import run_inference as dinf

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# return dataset for decoder
def initial_classifier_train(output_dir, train_dataset, test_dataset, depth):
	print "INITIAL CLASSIFIER TRAINING"
	tp = ctpd(output_dir, train_dataset, test_dataset, depth, num_epochs=1000, prefix="classifier_0")
	ctl(tp)
	results_dirs = get_immediate_subdirectories(output_dir)
	for results_dir in results_dirs:
		if len(results_dir)>12:
			if results_dir[0:12] == "classifier_0":
				return get_cd(os.path.join(output_dir, results_dir))

def decoder_train(output_dir, train_dataset, test_dataset, depth):
	print "DECODER TRAINING"
	num_decoder_dirs = 0
	results_dirs = get_immediate_subdirectories(output_dir)
	for results_dir in results_dirs:
		if len(results_dir)>7:
			if results_dir[0:7] == "decoder":
				num_decoder_dirs+=1
	prefix_len = 8+len(str(num_decoder_dirs))

	# run training
	tp = dtpd(output_dir, train_dataset, test_dataset, depth,
				num_epochs=1000, prefix="decoder_"+str(num_decoder_dirs))
	dtl(tp)

	# get results dir
	results_dirs = get_immediate_subdirectories(output_dir)
	for results_dir in results_dirs:
		if len(results_dir)>prefix_len:
			if results_dir[0:prefix_len] == "decoder_"+str(num_decoder_dirs):
				decoder_results_dir = os.path.join(output_dir, results_dir)

	# run inference on s points
	model = dlm(decoder_results_dir)
	train_s_points = train_dataset["s_points"]
	test_s_points = test_dataset["s_points"]
	train_s_inputs = dinf(train_s_points, model)
	test_s_inputs = dinf(test_s_points, model)

	train_s_dataset = {}
	test_s_dataset = {}
	train_s_dataset["X"] = train_s_inputs
	test_s_dataset["X"] = test_s_inputs
	train_s_dataset["Y"] = train_dataset["s_labels"]
	test_s_dataset["Y"] = test_dataset["s_labels"]
	return [train_s_dataset, test_s_dataset]

def classifier_train(output_dir, train_dataset, test_dataset, train_s_dataset, test_s_dataset, depth):
	print "CLASSIFIER TRAINING"
	num_classifier_dirs = 0
	results_dirs = get_immediate_subdirectories(output_dir)
	for results_dir in results_dirs:
		if len(results_dir)>10:
			if results_dir[0:10] == "classifier":
				num_classifier_dirs+=1
	prefix_len = 11+len(str(num_classifier_dirs))

	# run training
	tp = ctpd(output_dir, train_dataset, test_dataset, depth,
			train_s_dataset=train_s_dataset, test_s_dataset=test_s_dataset,
			num_epochs=1000, prefix = "classifier_"+str(num_classifier_dirs))
	ctl(tp)

	# get results dir
	results_dirs = get_immediate_subdirectories(output_dir)
	for results_dir in results_dirs:
		if len(results_dir)>prefix_len:
			if results_dir[0:prefix_len] == "classifier_"+str(num_classifier_dirs):
				return get_cd(os.path.join(output_dir, results_dir))

def full_loop(output_dir, input_data_file, depth, num_iters):
	train_cd, test_cd = pttd(input_data_file)

	train_dd, test_dd = initial_classifier_train(output_dir, train_cd, test_cd, depth)

	for i in range(num_iters):
		train_sd, test_sd = decoder_train(output_dir, train_dd, test_dd, depth)
		train_dd, test_dd = classifier_train(output_dir, train_cd, test_cd, train_sd, test_sd, depth)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output_dir')
	parser.add_argument('-i', '--input_data')
	parser.add_argument('-d', '--depth', type=int, default=2)
	parser.add_argument('-t', '--iterations', type=int, default=1)
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	full_loop(args.output_dir, args.input_data, args.depth, args.iterations)
