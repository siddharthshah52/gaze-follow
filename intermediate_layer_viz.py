import caffe
import matplotlib.pyplot as plt
import pdb
import skimage.io as io
import math
import numpy as np
from skimage.transform import resize
import scipy.io
import Image
import skimage.color as color
import csv
import lmdb
import cv2
import pandas as pd

caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')

datum = caffe.proto.caffe_pb2.Datum()
phase = 'test'

#caffe net
caffe.set_mode_cpu()

net1 = caffe.Net('./train.prototxt', './snapshot__iter_300.caffemodel', caffe.TEST)
net2 = caffe.Net(caffe_root + 'examples/gazefollow_model/deploy_demo.prototxt', caffe_root + 'examples/gazefollow_model/binary_w.caffemodel', caffe.TEST)
# print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

#mean images
places_mean = scipy.io.loadmat(caffe_root + 'examples/gazefollow_model/places_mean_resize.mat')
imagenet_mean = scipy.io.loadmat(caffe_root + 'examples/gazefollow_model/imagenet_mean_resize.mat')

mean_img_bgr = places_mean['image_mean'].transpose((2,1,0))
mean_face_bgr = imagenet_mean['image_mean'].transpose((2,1,0))

annotations = open('./annotations_final_test.csv', 'rb')
reader = csv.reader(annotations)
df = pd.read_csv('./annotations_final_test.csv')
df = df.drop_duplicates(subset='filename')

env_head = lmdb.open('%s_head_lmdb' % phase, readonly=True)
env_image = lmdb.open('%s_image_lmdb' % phase, readonly=True)
env_eye = lmdb.open('%s_eye_lmdb' % phase, readonly=True)

txn_head = env_head.begin()
txn_image = env_image.begin()
txn_eye = env_eye.begin()

i = 0
reader.next()
for key, _ in txn_image.cursor():
  line = reader.next()
  # print line
  # print df.iloc[i]
  net1.blobs['data'].reshape(1, 3, 227, 227)
  net1.blobs['face'].reshape(1, 3, 227, 227)
  net1.blobs['eyes_grid'].reshape(1, 169, 1, 1)

  net2.blobs['data'].reshape(1, 3, 227, 227)
  net2.blobs['face'].reshape(1, 3, 227, 227)
  net2.blobs['eyes_grid'].reshape(1, 169, 1, 1)

  raw_im = txn_image.get(key)
  datum = caffe.proto.caffe_pb2.Datum()
  datum.ParseFromString(raw_im)
  flat_im = np.fromstring(datum.data, dtype=np.uint8)
  im = flat_im.reshape(datum.channels, datum.height, datum.width)
  y = datum.label
  # print y

  image = im.transpose(1,2,0)
  image = image[:,:,[2,1,0]]

  plt.suptitle("Model output comparison")
  ax1 = plt.subplot(131)
  ax1.set_title("Original Image")
  plt.imshow(image)
  plt.axis('off')
  # plt.show()

  im = im - mean_img_bgr

  net1.blobs['data'].data[...] = im
  net2.blobs['data'].data[...] = im

  raw_head = txn_head.get(key)
  datum = caffe.proto.caffe_pb2.Datum()
  datum.ParseFromString(raw_head)
  flat_head = np.fromstring(datum.data, dtype=np.uint8)
  head = flat_head.reshape(datum.channels, datum.height, datum.width)

  face = head.transpose(1,2,0)
  face = face[:,:,[2,1,0]]

  net1.blobs['face'].data[...] = head
  net2.blobs['face'].data[...] = head

  head = head - mean_face_bgr

  raw_eye = txn_eye.get(key)
  datum = caffe.proto.caffe_pb2.Datum()
  datum.ParseFromString(raw_eye)
  flat_eye = np.fromstring(datum.data, dtype=np.uint8)
  eye = flat_eye.reshape(datum.channels, datum.height, datum.width)

  eye = eye[:, :, :, np.newaxis]
  net1.blobs['eyes_grid'].data[...] = eye
  net2.blobs['eyes_grid'].data[...] = eye

  net1.forward()
  net2.forward()

  print 'gaze', int(df.iloc[i]['gaze_x']*13), int(df.iloc[i]['gaze_y']*13 )

  new_output = net1.blobs['conv_reduce_reshape'].data
  new_output = new_output.reshape((13,13))
  print 'our_pred', np.unravel_index(np.argmax(new_output), new_output.shape)
  ax2 = plt.subplot(133)
  ax2.set_title("Our model")
  plt.imshow(new_output)
  plt.axis('off')
  # plt.show()

  output = net2.blobs['fc_7'].data[0, 0]
  print 'their_pred', np.unravel_index(np.argmax(output), output.shape)
  ax3 = plt.subplot(132)
  ax3.set_title('Their model')
  plt.imshow(output.reshape((13,13)))
  plt.axis('off')
  plt.show()

  i += 1
  if i == 6:
    break

# pdb.set_trace()