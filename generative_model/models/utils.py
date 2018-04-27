import sys

def assert_key_in_dict(key, dict_):
	if not key in dict_:
		print("key "+key+" required in dict")
		sys.exit()
