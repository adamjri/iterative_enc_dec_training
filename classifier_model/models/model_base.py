try:
	from classifier_model.models.utils import convert_to_string, write_dataset_to_file
except:
	try:
		from models.utils import convert_to_string, write_dataset_to_file
	except:
		from utils import convert_to_string, write_dataset_to_file

from abc import ABCMeta, abstractmethod, abstractproperty
from keras.callbacks import Callback
from keras.utils import plot_model

import numpy as np
import json

import datetime
import os
import sys

def shuffle_inputs_labels(inputs, labels):
	nums = np.array(list(range(len(inputs))))
	np.random.shuffle(nums)
	inputs = inputs[nums]
	lables = labels[nums]

class TestResultsCallback(Callback):

	def __init__(self, model, test_dataset, test_batch_size, epochs_per_test,
	 			results_names, train_dir=None, log_file=None,
				num_test_vis=None, visualizer=None, test_s_dataset=None, **kwargs):

		self.model = model
		self.test_dataset = test_dataset
		self.test_data = np.array(test_dataset["X"])
		self.test_labels = np.array(test_dataset["Y"])
		self.full_test_data = self.test_data
		self.full_test_labels = self.test_labels
		if not test_s_dataset is None:
			self.full_test_data = np.concatenate((self.full_test_data, np.array(test_s_dataset["X"])))
			self.full_test_labels = np.concatenate((self.full_test_labels, np.array(test_s_dataset["Y"])))

		self.test_batch_size = test_batch_size
		self.epochs_per_test = epochs_per_test
		self.results_names = results_names
		self.train_dir = train_dir
		self.log_file = log_file
		self.num_test_vis = num_test_vis
		self.visualizer = visualizer
		self.visualizer_inputs = kwargs

	def on_train_begin(self, logs={}):
		self.num_tests = 0
		self.test_results = []
		self.accuracies = []
		if not self.log_file is None:
			header = "\"epoch\", "
			for name in self.results_names:
				header+="\""+name+"\", "
			header += "\"accuracy\""
			f = open(self.log_file, 'w')
			f.write(header+"\n")
			f.close()

	def on_epoch_end(self, epoch, logs={}):
		if (epoch+1)%self.epochs_per_test==0:

			results = self.model.evaluate(self.full_test_data, self.full_test_labels,
										batch_size=self.test_batch_size,
										verbose=0)
			if type(results) != list:
				results = [results]

			results.insert(0, epoch)
			self.test_results.append(results)

			# get classification accuracy
			predictions = self.model.predict(self.test_data, batch_size=self.test_batch_size, verbose=0)
			num_total = len(self.test_labels)
			num_correct = 0
			for i in range(num_total):
				is_one = predictions[i][0]>=0.5
				label_one = self.test_labels[i][0]==1
				if is_one==label_one:
					num_correct+=1
			accuracy = float(num_correct)/float(num_total)
			self.accuracies.append(accuracy)

			if not self.log_file is None:
				results_str = ""
				for i in range(len(results)):
					results_str+=convert_to_string(results[i])+", "
				results_str+=convert_to_string(accuracy)
				f = open(self.log_file, 'a')
				f.write(results_str+"\n")
				f.close()

			if not self.train_dir is None:
				epoch_dir = os.path.join(self.train_dir, "Epoch_"+str(epoch))
				os.makedirs(epoch_dir)
				model_file = os.path.join(epoch_dir, "model.h5")
				self.model.save(model_file)

				if not self.num_test_vis is None:
					if (self.num_tests+1)%self.num_test_vis==0:
						if not self.visualizer is None:
							self.visualizer(self.model, self.test_dataset, self.test_batch_size,
							 				epoch_dir, **self.visualizer_inputs)

			self.num_tests += 1

class TrainLossCallback(Callback):
	def __init__(self, log_file=None):
		self.log_file = log_file

	def on_train_begin(self, logs={}):
		self.losses = []
		if not self.log_file is None:
			header = "\"loss\""
			f = open(self.log_file, 'w')
			f.write(header+"\n")
			f.close()

	def on_batch_end(self, batch, logs={}):
		l = logs.get('loss')
		self.losses.append(l)
		if not self.log_file is None:
			# clear file
			f = open(self.log_file, 'a')
			f.write(convert_to_string(l)+"\n")
			f.close()

class ModelBase():
	__metaclass__ = ABCMeta

	def __init__(self, params_dict):
		self.params = params_dict
		self.verify_params()
		self.model = None

	@abstractmethod
	def verify_params(self):
		pass

	@abstractmethod
	def load_model(self):
		pass

	def train(self, train_dataset, batch_size,
			test_dataset, test_batch_size, epochs, epochs_per_test,
			num_test_vis=None, visualizer=None,
			output_dir=None, train_s_dataset=None, test_s_dataset=None,
			**kwargs):

		train_inputs = np.array(train_dataset["X"])
		train_labels = np.array(train_dataset["Y"])
		test_inputs = np.array(test_dataset["X"])
		test_labels = np.array(test_dataset["Y"])

		# append s datasets present
		if not train_s_dataset is None:
			train_inputs = np.concatenate((train_inputs, np.array(train_s_dataset["X"])))
			train_labels = np.concatenate((train_labels, np.array(train_s_dataset["Y"])))
			shuffle_inputs_labels(train_inputs, train_labels)

		train_params = {}
		train_params["sample_size"] = len(train_inputs)+len(test_inputs)
		train_params["train_sample_size"] = len(train_inputs)
		train_params["test_sample_size"] = len(test_inputs)
		train_params["num_epochs"] = epochs
		train_params["batch_size"] = batch_size
		train_params["num_epochs_per_test"] = epochs_per_test
		train_params["test_batch_size"] = test_batch_size

		serializable_model_params = {}
		for p in self.params:
			try:
				p_str = json.dumps(self.params[p])
			except:
				p_str = self.params[p].__class__.__name__
			finally:
				serializable_model_params[p] = p_str


		# create log files
		timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
		if "prefix" in kwargs:
			timestamp = kwargs["prefix"]+"_"+timestamp
		train_dir = os.path.join(output_dir, timestamp)
		train_log_file = None
		test_log_file = None
		model_params_file = None
		train_params_file = None
		train_data_file = None
		test_data_file = None
		if not output_dir is None:
			os.makedirs(train_dir)
			train_log_file = os.path.join(train_dir, "train_log.txt")
			test_log_file = os.path.join(train_dir, "test_log.txt")

			model_params_file = os.path.join(train_dir, "model_params.txt")
			f = open(model_params_file, 'w')
			f.write(json.dumps(serializable_model_params))
			f.close()

			train_params_file = os.path.join(train_dir, "train_params.txt")
			f = open(train_params_file, 'w')
			f.write(json.dumps(train_params))
			f.close()

			train_data_file = os.path.join(train_dir, "train_data.txt")
			test_data_file = os.path.join(train_dir, "test_data.txt")
			write_dataset_to_file(train_data_file, train_dataset)
			write_dataset_to_file(test_data_file, test_dataset)

			model_file = os.path.join(train_dir, "model.png")
			plot_model(self.model, to_file=model_file, show_shapes=True)
			model_json_file = os.path.join(train_dir, "model.json")
			model_str = self.model.to_json()
			f = open(model_json_file, 'w')
			f.write(model_str)
			f.close()

		results_names = self.model.metrics_names

		train_callback = TrainLossCallback(log_file=train_log_file)
		test_callback = TestResultsCallback(self.model, test_dataset, test_batch_size, epochs_per_test,
											results_names, train_dir=train_dir, log_file=test_log_file,
											num_test_vis=num_test_vis, visualizer=visualizer,
											test_s_dataset=test_s_dataset, train_dataset=train_dataset,
											**kwargs)

		verbosity=2
		if "verbosity" in kwargs:
			verbosity = kwargs["verbosity"]
		self.model.fit(train_inputs, train_labels, batch_size, verbose=verbosity, epochs=epochs,
						callbacks=[train_callback, test_callback])
