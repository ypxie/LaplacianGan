# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

from inception.slim import slim
import numpy as np
import tensorflow as tf
from six.moves import urllib
import tarfile


import math
import os.path
import scipy.misc 
# import time
# import scipy.io as sio
# from datetime import datetime
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import json

tf.app.flags.DEFINE_string('checkpoint_dir',
                           './inception_finetuned_models/birds_valid299/model.ckpt',
                           """Path where to read model checkpoints.""")

tf.app.flags.DEFINE_string('image_folder', 
							'',
							"""Path where to load the images """)
tf.app.flags.DEFINE_string('h5_file', 
							'',
							"""Path where to load the images """)

tf.app.flags.DEFINE_integer('num_classes', 50,      # 20 for flowers
                            """Number of classes """)
tf.app.flags.DEFINE_integer('splits', 10,
                            """Number of splits """)
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('gpu', 0, "The ID of GPU to use")

FLAGS = tf.app.flags.FLAGS
# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

print ('*'*20)
fullpath = FLAGS.image_folder
print(fullpath)


def preprocess(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3),
                              interp='bilinear')
    img = img.astype(np.float32)
    # [0, 255] --> [0, 1] --> [-1, 1]
    img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return np.expand_dims(img, 0)



def load_data_from_h5(fullpath):
    import glob
    import deepdish as dd
    import h5py

    # import pdb; pdb.set_trace()
    h5file = os.path.join(fullpath, FLAGS.h5_file)
    return_path = os.path.join(fullpath, FLAGS.h5_file[:-3]+'_inception_score')
    print ('read h5 from {}'.format(h5file))

    
    #ms_images = {'output_64': [], 'output_128': [], 'output_256': [], }
    ms_images = {'output_64': [], }
    # ms_images = {'output_64': [], 'output_128': []}
    # assert len(ms_data.keys()) == 3, 'keys {}'.format(ms_data.keys())   

    for k in ms_images.keys():
        data = h5py.File(h5file)[k]
        images = []
        for i in range(data.shape[0]):
            img = data[i]
            # import pdb; pdb.set_trace()
            assert((img.shape[0] == 256 or img.shape[0] == 128 or img.shape[0] == 64) and img.shape[2] == 3)
            if not (img.min() >= 0 and img.max() <= 255 and img.mean() > 1):
                print ('WARNING {}, min {}, max {}, mean {}'.format(i, img.min(), img.max(), img.mean()))
                continue	
            #assert img.min() >= 0 and img.max() <= 255 and img.mean() > 1, '{}, min {}, max {}, mean {}'.format(i, img.min(), img.max(), img.mean())
            images.append(img)
        print ('read {} with {} images'.format(k, data.shape[0]))
        ms_images[k] = images

    print ('Totally {} images/scale are loaded at scales {}'.format(len(images), ms_images.keys() ))

    return ms_images, return_path

MODEL_DIR = './inception_finetuned_models/imagenet'

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 100
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        #sys.stdout.write(".")
        #sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
        print("%d of %d batches" % (i, n_batches)),
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)
    
def init_inception_model(sess, num_class):
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)

    # graph = tf.Graph()
    # with graph.as_default():

    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    var = None
    with tf.gfile.FastGFile(os.path.join(
        MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        var = tf.import_graph_def(graph_def, name='')
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    with sess.as_default():
        INPUT = sess.graph.get_tensor_by_name('ExpandDims:0')

        # weight = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        # logits = tf.matmul(tf.squeeze(pool3, axis=[0,1]), weight) 
        # softmax = tf.nn.softmax(logits)

        ## use a new softmax
        model_vars = tf.global_variables()
        weight = tf.Variable(tf.random_normal([2048, num_class], stddev=0.02))
        LOGITS = tf.matmul(tf.squeeze(pool3, axis=[1,2]), weight)   
         
        return INPUT, LOGITS, model_vars



  
if softmax is None:
    _init_inception()

ms_images, return_save_path = load_data_from_h5(fullpath)
#ms_images, return_save_path = load_data_from_h5_fakehr(fullpath)
ms_means = {k:[] for k in ms_images.keys()}
ms_std = {k:[] for k in ms_images.keys()}
for scale, images in ms_images.items():
    mean, std = get_inception_score(images)
    ms_means[scale] = float(mean)
    ms_std[scale] = float(std)
    print ('scale: {} mean: {} std:{}'.format(scale, mean, std))
print (ms_means, ms_std)
json.dump({'mean': ms_means, 'std': ms_std}, open(return_save_path + '.json','w'), indent=True)
