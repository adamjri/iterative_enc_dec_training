try:
	from classifier_model.utils import load_dataset_from_file
except:
	from utils import load_dataset_from_file

import random
import numpy as np
import sys

def parse_train_test_datasets(dataset_file, train_ratio=0.85):
	dataset, header = load_dataset_from_file(dataset_file)
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
