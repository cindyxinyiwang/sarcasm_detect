import numpy as np 

neg = np.load("postrain.npy")
l = len(neg)

i_l = l / 10

count = 0
c_line = 0
file_name = "pos" + str(count) + ".txt"
out_file = open(file_name, "w")
for line in neg:
	if c_line > i_l:
		count += 1
		file_name = "pos" + str(count) + ".txt"
		c_line = 0
		out_file = open(file_name, "w")
	out_file.write(line.decode("utf-8"))
	out_file.write('\n')
	c_line += 1