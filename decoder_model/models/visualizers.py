import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
from keras import backend as K

try:
	from decoder_model.models.utils import write_dataset_to_file
except:
	try:
		from models.utils import write_dataset_to_file
	except:
		from utils import write_dataset_to_file


def simple_dense_visualizer(model, test_dataset, test_batch_size, output_dir, **kwargs):

	has_separator = 'separator' in test_dataset

	inp = model.input
	outputs = [layer.output for layer in model.layers]
	functor = K.function([inp] + [K.learning_phase()], outputs )

	# get outputs from each layer from test dataset
	input_data = np.array(test_dataset["X"])
	layer_outs = functor([input_data, 1.])
	num_layers = len(layer_outs)

	if "train_dataset" in kwargs:
		train_input_data = np.array(kwargs["train_dataset"]["X"])
		train_layer_outs = functor([train_input_data, 1.])

	if has_separator:
		separator_data = np.array(test_dataset["separator"])
		separator_outs = functor([separator_data, 1.])

	# iterate over layers
	for i in range(num_layers):
		layer_dir = os.path.join(output_dir, "layer_"+str(i))
		os.makedirs(layer_dir)
		dim = len(layer_outs[i][0])
		values = layer_outs[i]

		# write data to file
		layer_dataset = { "test_values": values }
		if "train_dataset" in kwargs:
			layer_dataset["train_values"] = train_layer_outs[i]
		if has_separator:
			layer_dataset["separator"] = separator_outs[i]
		layer_filename = os.path.join(layer_dir, "outputs.txt")
		write_dataset_to_file(layer_filename, layer_dataset)

		# establish bounds for axis
		mins = np.array([min(values.T[j]) for j in range(dim)])
		maxs = np.array([max(values.T[j]) for j in range(dim)])
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

		if dim == 2:
			data_X_list = values.T[0]
			data_Y_list = values.T[1]

			plt.clf()

			plt.plot(data_X_list[j], data_Y_list[j], marker='o', c='b', label = "Values")

			if has_separator:
				sdata_X_list = separator_outs[i].T[0]
				sdata_Y_list = separator_outs[i].T[1]
				plt.plot(sdata_X_list[j], sdata_Y_list[j], marker='*', c='r', label="Separator")
			plt.legend(loc='best')
			plt.xlim(mins[0], maxs[0])
			plt.ylim(mins[1], maxs[1])
			plt.savefig(os.path.join(layer_dir, "2D_layer_"+str(i)))

		elif dim == 3:
			data_X_list = values.T[0]
			data_Y_list = values.T[1]
			data_Z_list = values.T[2]

			plt.clf()
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			ax.scatter(data_X_list, data_Y_list, data_Z_list, marker='o', c='b')

			if has_separator:
				sdata_X_list = separator_outs[i].T[0]
				sdata_Y_list = separator_outs[i].T[1]
				sdata_Z_list = separator_outs[i].T[2]

				ax.scatter(sdata_X_list[j], sdata_Y_list[j], sdata_Z_list[j], marker='o', c='r')

			ax.set_xlim3d(mins[0], maxs[0])
			ax.set_ylim3d(mins[1], maxs[1])
			ax.set_zlim3d(mins[2], maxs[2])
			plt.savefig(os.path.join(layer_dir, "3D_layer_"+str(i)))
			plt.close(fig)

		else:
			layer_file = os.path.join(layer_dir, "info.txt")
			f = open(layer_file, 'w')
			f.write("Layer is dimension "+str(dim)+". Cannot plot.")
			f.close()
