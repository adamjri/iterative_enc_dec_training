import numpy as np

import sys

def transform(h_mat, vec):
	aug_vec = np.concatenate((vec, np.array([1.0])))
	trans_aug_vec = h_mat.dot(aug_vec)
	return trans_aug_vec[:-1]

def rotate(h_mat, vec):
	dim = len(vec)
	simp_h_mat = h_mat[:dim, :dim]
	return simp_h_mat.dot(vec)

def assert_key_in_dict(key, dict_):
	if not key in dict_:
		print("key "+key+" required in dict")
		sys.exit()
