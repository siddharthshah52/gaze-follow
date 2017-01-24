import numpy as np
import cv2
import glob
import csv
import re

path = './gazefollow_data/'
opencv_cascades_path = '/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/'
filenames = glob.glob(path + 'train/*/*.jpg')

# Store annotations as comma separated values. The format is filepath, x, y, w, h
annotation_file = open('annotations.csv', 'ab')

writer = csv.writer(annotation_file)

lbpside = cv2.CascadeClassifier(opencv_cascades_path + 'lbpcascades/lbpcascade_profileface.xml')
lbpfront = cv2.CascadeClassifier(opencv_cascades_path + 'lbpcascades/lbpcascade_frontalface.xml')

haarside = cv2.CascadeClassifier(opencv_cascades_path + 'haarcascades/haarcascade_profileface.xml')
haarfront = cv2.CascadeClassifier(opencv_cascades_path + 'haarcascades/haarcascade_frontalface_default.xml')

if lbpfront.empty() or lbpside.empty() or haarfront.empty() or haarside.empty():
	print "cascades not loaded correctly"
	exit()


def find_face(cascade):
	faces = []

	for scale in [float(i)/10 for i in range(11, 15)]:
	    for neighbors in range(2,5):
	        temp = cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(20, 20))
	        # print 'scale: %s, neighbors: %s, len rects: %d' % (scale, neighbors, len(temp))
	        if len(temp)>0:
	        	faces = temp
	        	return faces
	return faces

i = 0

for filename in filenames:
	print filename[(len(path) + len('train')):]
	img = cv2.imread(filename)
	# print img

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces_front = find_face(lbpfront)
	if len(faces_front) == 0:
		faces_front = find_face(haarfront)

	faces_side = find_face(lbpside)
	if len(faces_side) == 0:
		faces_side = find_face(haarside)

	for (x,y,w,h) in faces_front:
		x -= 0.25*w
		y -= 0.25*h
		w *= 1.5
		h *= 1.5
		x = max(0, x)
		y = max(0, y)
		w = min(x+w, img.shape[1]) - x
		h = min(y+h, img.shape[0]) - y

		# use int values for x, y, w, h if you want to draw rectangles
		# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		writer.writerow([filename[len(path):], str(x/img.shape[1]), str(y/img.shape[0]), str(w/img.shape[1]), str(h/img.shape[0])])

	for (x,y,w,h) in faces_side:
		x -= 0.25*w
		y -= 0.25*h
		w *= 1.5
		h *= 1.5
		x = max(0, x)
		y = max(0, y)
		w = min(x+w, img.shape[1]) - x
		h = min(y+h, img.shape[0]) - y

		# use int values for x, y, w, h if you want to draw rectangles
		# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		writer.writerow([filename[len(path):], str(x/img.shape[1]), str(y/img.shape[0]), str(w/img.shape[1]), str(h/img.shape[0])])

	# cv2.imshow('img', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# i += 1
	# if i>4:
	# 	break