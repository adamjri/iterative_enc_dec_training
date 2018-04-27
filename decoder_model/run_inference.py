import numpy as np
import os
import sys

from keras.models import model_from_json

try:
	from decoder_model.utils import load_dataset_from_file
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

def run_inference(points, model):
	predictions = model.predict(np.array(points))
	return predictions.tolist()
