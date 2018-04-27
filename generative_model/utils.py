import numpy as np

import sys

def convert_to_string(var):
	output_str = ""
	# check if dict
	if isinstance(var, dict):
		output_str+="{"
		for key in var:
			output_str+=convert_to_string(key)+": "
			output_str+=convert_to_string(var[key])+", "
		if len(output_str)>1:
			output_str = output_str[:-2]
		output_str+="}"
	# check if list
	elif isinstance(var, list):
		output_str+="["
		for i in range(len(var)):
			output_str+=convert_to_string(var[i])+", "
		if len(output_str)>1:
			output_str = output_str[:-2]
		output_str+="]"
	# check if numpy
	elif isinstance(var, np.ndarray):
		output_str+="np["
		for i in range(len(var)):
			output_str+=convert_to_string(var[i])+", "
		if len(output_str)>1:
			output_str = output_str[:-2]
		output_str+="]"
	# check if string
	elif isinstance(var, str):
		output_str = "\""+var+"\""
	else:
		try:
			output_str = str(var)
		except:
			print "Variable not serializable"
			sys.exit()
	return output_str.replace("\n", "")

def parse(input_str, delim):
	if len(input_str)==0:
		return []
	output = []
	l = len(delim)
	stack = []
	pointer = 0
	i = 0
	while True:
		if i>=len(input_str):
			output.append(input_str[pointer:])
			break

		if i<len(input_str)-l+1:
			if input_str[i:i+l]==delim:
				if len(stack)==0:
					output.append(input_str[pointer:i])
					pointer = i+l
					i+=l
					continue

		if i<len(input_str)-2:
			if input_str[i:i+3]=="np[":
				stack.append("[")
				i+=3
				continue
		if input_str[i]=="[":
			stack.append("[")
		elif input_str[i]=="{":
			stack.append("{")
		elif input_str[i]=="]":
			if stack[-1]=="[":
				stack = stack[:-1]
			else:
				print "Parsing invalid"
				sys.exit()
		elif input_str[i]=="}":
			if stack[-1]=="{":
				stack = stack[:-1]
			else:
				print "Parsing invalid"
				sys.exit()

		i+=1

	return output

def convert_from_string(var_str):
	# check if dict
	if var_str[0]=="{":
		output = {}
		if len(var_str)==2:
			return output
		elems = parse(var_str[1:-1], ", ")
		for elem in elems:
			k, v = parse(elem, ": ")
			key = convert_from_string(k)
			value = convert_from_string(v)
			output[key] = value
		return output
	# check if list
	if var_str[0]=="[":
		output = []
		if len(var_str)==2:
			return output
		elems = parse(var_str[1:-1], ", ")
		for elem in elems:
			output.append(convert_from_string(elem))
		return output
	# check if numpy
	if var_str[0:3]=="np[":
		output = []
		if len(var_str)==4:
			return np.array(output)
		elems = parse(var_str[3:-1], ", ")
		for elem in elems:
			output.append(convert_from_string(elem))
		return np.array(output)
	# check if string
	if var_str[0]=="\"":
		return var_str[1:-1]
	# check if bool
	if var_str=="True":
		return True
	if var_str=="False":
		return False
	# convert to int or float
	if "." in var_str:
		return float(var_str)
	else:
		return int(var_str)

def write_dataset_to_file(filename, dataset, header=None):
	f = open(filename, 'w')
	# Print header starting with #
	if not header is None:
		f.write("#"+header+"\n")
	# Print dataset keys
	keys = [key for key in dataset]
	if len(keys)==0:
		print "No keys in dataset"
		sys.exit()

	key_str = ""
	for key in keys:
		key_str+=convert_to_string(key)+", "
	if len(key_str)>0:
		key_str = key_str[:-2]
	f.write(key_str+"\n")

	key_lens = [len(dataset[key]) for key in keys]
	max_len = max(key_lens)
	for i in range(max_len):
		line_str = ""
		for j in range(len(keys)):
			if i>=key_lens[j]:
				line_str+=", "
			else:
				line_str+=convert_to_string(dataset[keys[j]][i])+", "
		f.write(line_str[:-2]+"\n")

	f.close()


def load_dataset_from_file(filename):
	f = open(filename, 'r')
	lines = f.readlines()
	start = 0
	header = None
	if lines[0][0]=="#":
		header = lines[0][1:-1]
		start+=1
	else:
		header = None

	dataset = {}
	keys = [convert_from_string(key_str) for key_str in parse(lines[start][:-1], ", ")]
	for key in keys:
		dataset[key] = []
	for line in lines[start+1:]:
		if len(line)==1:
			continue
		line_splt = parse(line[:-1], ", ")
		if not len(line_splt)==len(keys):
			print "File not properly formatted"
		for i in range(len(line_splt)):
			if len(line_splt[i])>0:
				dataset[keys[i]].append(convert_from_string(line_splt[i]))

	f.close()

	return [dataset, header]
