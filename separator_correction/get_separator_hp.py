import numpy as np
import os
import sys

from keras.models import model_from_json

try:
	from separator_correction.utils import load_dataset_from_file
except:
	from utils import load_dataset_from_file

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

def get_final_layer_weights_bias(model):
	weights, biases = model.layers[-1].get_weights()
	return [weights.T, biases]

# returns N and R, the normal and translation
def get_separator_hp_from_weights_biases(weights, biases):
	single_node = len(biases)==1
	if single_node:
		N = weights[0]
		norm = np.linalg.norm(N)
		N /= norm
		R = (-biases[0]/norm)*N
	else:
		u1 = weights[0]
		u2 = weights[1]
		N = u1 - u2
		norm = np.linalg.norm(N)
		N /= norm
		R = ((biases[1]-biases[0])/norm)*N
	return [N, R]

def get_separator_hp(results_dir, model_json_file = "model.json",
									model_h5_file = "model.h5",
									test_log_file = "test_log.txt"):
	model = load_model_from_results_dir(results_dir, model_json_file=model_json_file,
														model_h5_file=model_h5_file,
														test_log_file=test_log_file)
	w, b = get_final_layer_weights_bias(model)
	return get_separator_hp_from_weights_biases(w, b)


if __name__ == "__main__":
	results_dir = "/scratch/richards/generative_data/classifier_results/test_4/trial_0_20180228165725"
	print get_separator_hp(results_dir)
