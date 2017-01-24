import csv
import cv2
import skimage.io as sio
import numpy as np

f = open('annotations_final_test.csv', 'rb')

i = 0

reader = csv.reader(f)
l = list(reader)

np.random.shuffle(l)

for line in l:
	img = cv2.imread('../gazefollow_data/' + line[0])
	print line[0]

	h, w = img.shape[0], img.shape[1]
	vals = [float(x) for x in line[1:]]

	# cv2.imshow("Original", img)
	# cv2.waitKey(0)

	cv2.circle(img, (int(vals[-4]*w), int(vals[-3]*h)), 5, (0, 255, 0))
	cv2.circle(img, (int(vals[-2]*w), int(vals[-1]*h)), 5, (255, 0, 0))
	cv2.rectangle(img, (int(vals[0]*w), int(vals[1]*h)), (int((vals[0] + vals[2])*w), int((vals[1] + vals[3])*h)), (0,0, 255))
	cv2.imshow("BB", img)
	cv2.waitKey(0)

	if i==10:
		break
	i+=1

