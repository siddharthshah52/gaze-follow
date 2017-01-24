import csv
import numpy as np
import math

file = open('./annotations_final_train.csv')

reader = csv.reader(file)

shifts = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]

for line in reader:
	[gaze_x, gaze_y] = [float(line[-4]) * 15, float(line[-3]) * 15]
	encoding = np.zeros((5, 5, 5))
	for i, shift in enumerate(shifts):
		x = (gaze_x - shift[i][0]) // 3
		y = (gaze_y - shift[i][1]) // 3
		x = max(min(14, x), 0)
		y = max(min(14, y), 0)
		encoding[i, x, y] = 1 

