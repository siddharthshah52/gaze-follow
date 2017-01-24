__author__ = 'sumant'

import math

import caffe
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# CAFFE_ROOT = '/Users/sumant/code/caffe/'

# DATA_DIR = CAFFE_ROOT + 'data/gaze_follow/'

DATA_DIR = '/Users/siddharthshah/Acads/GT_Spring_2016/DL/Project/gazefollow_data/'

import pandas as pd
import lmdb
phase = 'test'
annot = pd.read_csv('annotations_final_%s.csv' % phase)

def write_image_to_lmdb(env, img, label, idx):
    with env.begin(write=True) as txn:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = img.shape[2]
        datum.height = img.shape[0]
        datum.width = img.shape[1]
        datum.data = img.transpose(2, 0, 1).tobytes()
        str_id = '{:08}'.format(idx)
        datum.label = int(label)
        txn.put(str_id, datum.SerializeToString())

size=1*1024*1024*1024*10

for i in range(0, min(annot.values.shape[0], 20000)):
    env_img = lmdb.open('%s_image_lmdb' % phase, map_size=size)
    env_head = lmdb.open('%s_head_lmdb' % phase, map_size=size)
    env_eye = lmdb.open('%s_eye_lmdb' % phase, map_size=size)

    row = annot.values[i]
    file_name = str(row[0])

    eye_arr = np.zeros((169, 1, 1), dtype=np.uint8)

    eye_row = int(row[7]*13)
    eye_col = int(row[8]*13)
    eye_arr[13*eye_row + eye_col] = 1

    image = cv2.imread(DATA_DIR + file_name)
    r = int(image.shape[0] * row[2])
    c = int(image.shape[1] * row[1])
    r_offset = int(image.shape[0] * row[4])
    c_offset = int(image.shape[1] * row[3])
    head_image = image[r:r+r_offset, c:c+c_offset, :]

    gaze_r = int(13 * row[6])
    gaze_c = int(13 * row[5])
    gaze_location = gaze_r * 13 + gaze_c

    image = cv2.resize(image, (227, 227))
    head_image = cv2.resize(head_image, (227, 227))
    write_image_to_lmdb(env_head, head_image, gaze_location, i)
    write_image_to_lmdb(env_img, image, gaze_location, i)
    write_image_to_lmdb(env_eye, eye_arr, gaze_location, i)
    print i


env = lmdb.open('%s_head_lmdb' % phase, readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

import numpy as np
flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.channels, datum.height, -1)
y = datum.label

plt.imshow(x.transpose(1, 2, 0))