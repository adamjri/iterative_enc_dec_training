try:
	from generative_model.plotting.utils import load_dataset_from_file
except:
	try:
		from plotting.utils import load_dataset_from_file
	except:
		from utils import load_dataset_from_file

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def plot_2d_dataset_colored(dataset, separator=None, savefile=None):
	if not len(dataset["X"][0])==2:
		print "Dataset input not 2D"
		sys.exit()
	if not len(dataset["Y"][0])==1:
		print "Dataset output not 1D"
		sys.exit()

	X_list_0 = [dataset["X"][i][0] for i in range(len(dataset["X"])) if dataset["Y"][i][0]<0.5]
	Y_list_0 = [dataset["X"][i][1] for i in range(len(dataset["X"])) if dataset["Y"][i][0]<0.5]
	X_list_1 = [dataset["X"][i][0] for i in range(len(dataset["X"])) if dataset["Y"][i][0]>=0.5]
	Y_list_1 = [dataset["X"][i][1] for i in range(len(dataset["X"])) if dataset["Y"][i][0]>=0.5]

	x_min = min(min(X_list_0), min(X_list_1))
	x_max = max(max(X_list_0), max(X_list_1))
	y_min = min(min(Y_list_0), min(Y_list_1))
	y_max = max(max(Y_list_0), max(Y_list_1))
	x_dist = x_max - x_min
	y_dist = y_max - y_min
	if x_dist==0 and y_dist==0:
		x_dist = 10.0
		y_dist = 10.0
	if x_dist == 0:
		x_dist = y_dist
	elif y_dist == 0:
		y_dist = x_dist
	x_min = x_min - x_dist/10.0
	x_max = x_max + x_dist/10.0
	y_min = y_min - y_dist/10.0
	y_max = y_max + y_dist/10.0

	plt.clf()
	plt.plot(X_list_0, Y_list_0, 'ro', label="Red: 0")
	plt.plot(X_list_1, Y_list_1, 'go', label="Green: 1")
	if not separator is None:
		sep_X = [sep[0] for sep in separator]
		sep_Y = [sep[1] for sep in separator]
		plt.plot(sep_X, sep_Y, 'b*', label="Blue: separator")
	plt.legend(loc='best')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)

	if not savefile is None:
		plt.savefig(savefile)
	plt.show()

def plot_3d_dataset_colored(dataset, separator=None, savefile=None):
	if not len(dataset["X"][0])==3:
		print "Dataset input not 3D"
		sys.exit()
	if not len(dataset["Y"][0])==1:
		print "Dataset output not 1D"
		sys.exit()

	X_list_0 = [dataset["X"][i][0] for i in range(len(dataset["X"])) if dataset["Y"][i][0]<0.5]
	Y_list_0 = [dataset["X"][i][1] for i in range(len(dataset["X"])) if dataset["Y"][i][0]<0.5]
	Z_list_0 = [dataset["X"][i][2] for i in range(len(dataset["X"])) if dataset["Y"][i][0]<0.5]
	X_list_1 = [dataset["X"][i][0] for i in range(len(dataset["X"])) if dataset["Y"][i][0]>=0.5]
	Y_list_1 = [dataset["X"][i][1] for i in range(len(dataset["X"])) if dataset["Y"][i][0]>=0.5]
	Z_list_1 = [dataset["X"][i][2] for i in range(len(dataset["X"])) if dataset["Y"][i][0]>=0.5]

	x_min = min(min(X_list_0), min(X_list_1))
	x_max = max(max(X_list_0), max(X_list_1))
	y_min = min(min(Y_list_0), min(Y_list_1))
	y_max = max(max(Y_list_0), max(Y_list_1))
	z_min = min(min(Z_list_0), min(Z_list_1))
	z_max = max(max(Z_list_0), max(Z_list_1))
	x_dist = x_max - x_min
	y_dist = y_max - y_min
	z_dist = z_max - z_min
	if x_dist==0 and y_dist==0 and z_dist==0:
		x_dist = 50.0
		y_dist = 50.0
		z_dist = 50.0
	if x_dist == 0:
		x_dist = max(y_dist, z_dist)
	if y_dist == 0:
		y_dist = max(x_dist, z_dist)
	if z_dist == 0:
		z_dist = max(x_dist, y_dist)
	x_min = x_min - x_dist/10.0
	x_max = x_max + x_dist/10.0
	y_min = y_min - y_dist/10.0
	y_max = y_max + y_dist/10.0

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X_list_0, Y_list_0, Z_list_0, c='r', marker='o')
	ax.scatter(X_list_1, Y_list_1, Z_list_1, c='g', marker='o')
	if not separator is None:
		sep_X = [sep[0] for sep in separator]
		sep_Y = [sep[1] for sep in separator]
		sep_Z = [sep[2] for sep in separator]
		ax.scatter(sep_X, sep_Y, sep_Z, c='b', marker='o')
	# ax.set_xlim3d(x_min, x_max)
	# ax.set_ylim3d(y_min, y_max)
	# ax.set_zlim3d(z_min, z_max)
	if not savefile is None:
		plt.savefig(savefile)
	plt.show()
	plt.close(fig)





if __name__ == "__main__":
	separator = []
	for i in range(20):
		for j in range(20):
			separator.append(np.array([0.0, i, j]))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plt.show()
