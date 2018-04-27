import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
from keras import backend as K

try:
	from classifier_model.models.utils import write_dataset_to_file
except:
	try:
		from models.utils import write_dataset_to_file
	except:
		from utils import write_dataset_to_file

def get_labels(results, dim):
	if dim==1:
		return (results >= 0.5).astype(int)
	else:
		new_results = []
		for i in range(len(results)):
			new_results.append([np.argmax(results[i])])
		return np.array(new_results)


def split_values_by_labels(values, labels, num_labels):
	labeled_values_list = [[] for i in range(num_labels)]
	for i in range(len(values)):
		labeled_values_list[labels[i][0]].append(values[i])
	for i in range(num_labels):
		labeled_values_list[i] = np.array(labeled_values_list[i])
	return labeled_values_list


def simple_dense_visualizer(model, test_dataset, test_batch_size, output_dir, **kwargs):

	has_separator = 'separator' in test_dataset

	output_dim = len(test_dataset["Y"][0])
	num_labels = max(output_dim, 2)

	inp = model.input
	outputs = [layer.output for layer in model.layers]
	functor = K.function([inp] + [K.learning_phase()], outputs )

	# get outputs from each layer from test dataset
	input_data = np.array(test_dataset["X"])
	layer_outs = functor([input_data, 1.])
	num_layers = len(layer_outs)-1
	labels = get_labels(layer_outs[-1], output_dim)
	layers_data_split = [split_values_by_labels(layer_outs[i], labels, num_labels)
							for i in range(num_layers)]

	num_labels = max(output_dim, 2)

	if "train_dataset" in kwargs:
		train_input_data = np.array(kwargs["train_dataset"]["X"])
		train_layer_outs = functor([train_input_data, 1.])

	if has_separator:
		separator_data = np.array(test_dataset["separator"])
		separator_outs = functor([separator_data, 1.])
		separator_labels = get_labels(separator_outs[-1], output_dim)
		separator_data_split = [split_values_by_labels(separator_outs[i], separator_labels, num_labels)
								for i in range(num_layers)]

	# iterate over layers
	for i in range(num_layers):
		layer_dir = os.path.join(output_dir, "layer_"+str(i))
		os.makedirs(layer_dir)
		input_dim = len(layer_outs[i][0])

		values = layer_outs[i]

		# write data to file
		layer_dataset = { "test_X": values, "test_Y": layer_outs[-1], "test_GT": test_dataset["Y"]}
		if "train_dataset" in kwargs:
			layer_dataset["train_X"] = train_layer_outs[i]
			layer_dataset["train_Y"] = train_layer_outs[-1]
			layer_dataset["train_GT"] = kwargs["train_dataset"]["Y"]
		if has_separator:
			layer_dataset["separator_X"] = separator_outs[i]
			layer_dataset["separator_Y"] = separator_outs[-1]
		layer_filename = os.path.join(layer_dir, "outputs_to_labels.txt")
		write_dataset_to_file(layer_filename, layer_dataset)

		# establish bounds for axis
		mins = np.array([min(values.T[j]) for j in range(input_dim)])
		maxs = np.array([max(values.T[j]) for j in range(input_dim)])
		diffs = maxs-mins
		degenerates = diffs < 0.00000000000000001
		mins = mins - diffs/5.0
		maxs = maxs + diffs/5.0

		if np.all(degenerates):
			f = open(os.path.join(layer_dir, "degenerate_layer_"+str(i)+".txt"), 'w')
			f.write("Data is degenerate")
			f.close()
			continue
		else:
			max_diff = max(diffs)
			for j, d in enumerate(degenerates):
				if d:
					mins[j] = -7.0*max_diff/10.0
					maxs[j] = 7.0*max_diff/10.0


		# create color wheels
		output_color = iter(cm.rainbow(np.linspace(0.0, 0.5, num_labels, endpoint=False)))
		if has_separator:
			separator_color = iter(cm.rainbow(np.linspace(0.5, 1.0, num_labels, endpoint=False)))

		if input_dim == 2:
			data_X_list = []
			data_Y_list = []
			for data_split in layers_data_split[i]:
				if len(data_split)>0:
					data_X_list.append((data_split.T)[0])
					data_Y_list.append((data_split.T)[1])
				else:
					data_X_list.append([])
					data_Y_list.append([])

			plt.clf()

			for j in range(num_labels):
				if len(data_X_list[j])>0:
					plt.plot(data_X_list[j], data_Y_list[j], marker='o', c=next(output_color),
																		label="Label "+str(j))

			if has_separator:
				sdata_X_list = []
				sdata_Y_list = []
				for sdata_split in separator_data_split[i]:
					if len(sdata_split)>0:
						sdata_X_list.append((sdata_split.T)[0])
						sdata_Y_list.append((sdata_split.T)[1])
					else:
						sdata_X_list.append([])
						sdata_Y_list.append([])

				for j in range(num_labels):
					if len(sdata_X_list[j])>0:
						plt.plot(sdata_X_list[j], sdata_Y_list[j], marker='*', c=next(separator_color),
																	label="Separator "+str(j))
			plt.legend(loc='best')
			plt.xlim(mins[0], maxs[0])
			plt.ylim(mins[1], maxs[1])
			plt.savefig(os.path.join(layer_dir, "2D_layer_"+str(i)))

		elif input_dim == 3:
			data_X_list = []
			data_Y_list = []
			data_Z_list = []
			for data_split in layers_data_split[i]:
				if len(data_split)>0:
					data_X_list.append((data_split.T)[0])
					data_Y_list.append((data_split.T)[1])
					data_Z_list.append((data_split.T)[2])
				else:
					data_X_list.append([])
					data_Y_list.append([])
					data_Z_list.append([])

			plt.clf()
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			for j in range(num_labels):
				if len(data_X_list[j])>0:
					ax.scatter(data_X_list[j], data_Y_list[j], data_Z_list[j], marker='o',
																				c=next(output_color))
			if has_separator:
				sdata_X_list = []
				sdata_Y_list = []
				sdata_Z_list = []
				for sdata_split in separator_data_split[i]:
					if len(sdata_split)>0:
						sdata_X_list.append((sdata_split.T)[0])
						sdata_Y_list.append((sdata_split.T)[1])
						sdata_Z_list.append((sdata_split.T)[2])
					else:
						sdata_X_list.append([])
						sdata_Y_list.append([])
						sdata_Z_list.append([])

				for j in range(num_labels):
					if len(sdata_X_list[j])>0:
						ax.scatter(sdata_X_list[j], sdata_Y_list[j], sdata_Z_list[j], marker='o',
																					c=next(separator_color))
			ax.set_xlim3d(mins[0], maxs[0])
			ax.set_ylim3d(mins[1], maxs[1])
			ax.set_zlim3d(mins[2], maxs[2])
			plt.savefig(os.path.join(layer_dir, "3D_layer_"+str(i)))
			plt.close(fig)

		else:
			for j in range(input_dim):
				output_color = iter(cm.rainbow(np.linspace(0.0, 0.5, num_labels, endpoint=False)))
				if has_separator:
					separator_color = iter(cm.rainbow(np.linspace(0.5, 1.0, num_labels, endpoint=False)))

				plt.clf()

				for k in range(num_labels):
					X_data = ((layers_data_split[i][k].T)[j]).tolist()
					Y_data = [k for m in range(len(X_data))]
					plt.plot(X_data, Y_data, marker='o', c=next(output_color), label="Label "+str(k))

					if has_separator:
						sX_data = ((separator_data_split[i][k].T)[j]).tolist()
						sY_data = [k for m in range(len(sX_data))]
						plt.plot(sX_data, sY_data, marker='o', c=next(separator_color),
																label="Label "+str(k))

				plt.legend(loc='best')
				plt.xlim(mins[j], maxs[j])
				plt.ylim(-1, num_labels+1)
				plt.savefig(os.path.join(layer_dir, str(input_dim)+"D_axis_"+str(j)+"_layer_"+str(i)))
