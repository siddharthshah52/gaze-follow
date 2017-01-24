__author__ = 'sumant, bhavishya'

import math

import caffe
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

CAFFE_ROOT = '/Users/siddharthshah/caffe-master/'

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
shifts = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]

for i in range(0, min(annot.values.shape[0], 12000)):
    env_fc1 = lmdb.open('%s_fc1_lmdb' % phase, map_size=size)
    env_fc2 = lmdb.open('%s_fc2_lmdb' % phase, map_size=size)
    env_fc3 = lmdb.open('%s_fc3_lmdb' % phase, map_size=size)
    env_fc4 = lmdb.open('%s_fc4_lmdb' % phase, map_size=size)
    env_fc5 = lmdb.open('%s_fc5_lmdb' % phase, map_size=size)

    row = annot.values[i]

    gaze_r = min(int(15 * row[6]),14)
    gaze_c = min(int(15 * row[5]),14)

    dummy_image = np.zeros((1, 1, 1))

    label = np.zeros(5)
    for ii,shift in enumerate(shifts):
        x = (gaze_r - shift[0]) / 3
        y = (gaze_c - shift[1]) / 3
        x = int(x)
        y = int(y)
        x = max(min(4, x), 0)
        y = max(min(4, y), 0)
        
        label[ii] = x*5 + y    

    write_image_to_lmdb(env_fc1, dummy_image, label[0], i)
    write_image_to_lmdb(env_fc2, dummy_image, label[1], i)
    write_image_to_lmdb(env_fc3, dummy_image, label[2], i)
    write_image_to_lmdb(env_fc4, dummy_image, label[3], i)
    write_image_to_lmdb(env_fc5, dummy_image, label[4], i)
    print i


# env = lmdb.open('fc1_lmdb', readonly=True)
# with env.begin() as txn:
#     raw_datum = txn.get(b'00000000')

# datum = caffe.proto.caffe_pb2.Datum()
# datum.ParseFromString(raw_datum)

# import numpy as np
# flat_x = np.fromstring(datum.data, dtype=np.uint8)
# x = flat_x.reshape(datum.channels, datum.height, -1)
# y = datum.label

# plt.imshow(x.transpose(1, 2, 0))
