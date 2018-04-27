import numpy as np
import os
import sys

def get_s_points(points, HP):
	s_points = []
	for i in range(len(points)):
		s_points.append(points[i] + HP[1] - (points[i].dot(HP[0]))*HP[0])
	return s_points

def get_s_label(single_node):
	if single_node:
		return np.array([0.5])
	else:
		return np.array([0.5, 0.5])

def get_correction_list(outputs, labels):
	single_node = len(outputs[0])==1
	# labelize outputs
	correction_list = []
	if single_node:
		for i in range(len(outputs)):
			out_is_1 = outputs[i][0]>=0.5
			label_is_1 = labels[i][0]==1
			correction_list.append(out_is_1==label_is_1)
	else:
		for i in range(len(outputs)):
			out_is_first = outputs[i][0]>=outputs[i][1]
			label_is_first = labels[i][0]>=labels[i][1]
			correction_list.append(out_is_first==label_is_first)

	return correction_list

def get_s_point_set(s_points, correction_list):
	new_s_points = []
	for i in range(len(s_points)):
		if correction_list[i]:
			new_s_points.append(s_points[i])
	return new_s_points

def get_corrected_points(points, s_points, correction_list):
	corrected_points = []
	for i in range(len(points)):
		if correction_list[i]:
			corrected_points.append(points[i])
		else:
			corrected_points.append(s_points[i])
	return corrected_points
