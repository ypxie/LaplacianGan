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


def get_inception_score(sess, images, pred_op):
    splits = FLAGS.splits
    # assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    bs = FLAGS.batch_size
    preds = []
    num_examples = len(images)
    n_batches = int(math.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)
    for i in range(n_batches):
        inp = []
        # print('i*bs', i*bs)
        for j in range(bs):
            if (i*bs + j) == num_examples:
                break
            img = images[indices[i*bs + j]]
            # print('*****', img.shape)
            img = preprocess(img)
            inp.append(img)
        print("%d of %d batches" % (i, n_batches)),
        sys.stdout.flush()
        # inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        #  print('inp', inp.shape)
        pred = sess.run(pred_op, {'inputs:0': inp})
        preds.append(pred)
        # if i % 100 == 0:
        #     print('Batch ', i)
        #     print('inp', inp.shape, inp.max(), inp.min())
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part * (np.log(part) -
              np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    # print('mean:', "%.2f" % np.mean(scores), 'std:', "%.2f" % np.std(scores))
    return np.mean(scores), np.std(scores)


def load_data(fullpath):
    print(fullpath)
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                # print('filename', filename)
                # print('path', path, '\nname', name)
                # print('filename', filename)
                if os.path.isfile(filename):
                    img = scipy.misc.imread(filename)
                    images.append(img)
    print('images', len(images), images[0].shape)
    return images

def load_data_from_h5(fullpath):
    import glob
    import deepdish as dd
    import h5py

    # import pdb; pdb.set_trace()
    
    h5file = os.path.join(fullpath, FLAGS.h5_file)
    return_path = os.path.join(fullpath, FLAGS.h5_file[:-3]+'_inception_score')
    print ('read h5 from {}'.format(h5file))
    assert(os.path.isfile(h5file))
    fh = h5py.File(h5file)
    keys = [a for a in fh.keys() if 'output' in a]
    if 'output_512' in keys:
        ms_images = {'output_512': [], 'output_256': [], }
    elif 'output_256' in keys:
        ms_images = {'output_256': [], }
    elif 'output_128' in keys:
        ms_images = {'output_128': [], }
    elif 'output_64' in keys:
        ms_images = {'output_64': [], }
    # ms_images = {'output_64': [], 'output_128': [], 'output_256': [], }
    # ms_images = {'output_256': [], }
    # ms_images = {'output_512': [], }
    # assert len(ms_data.keys()) == 3, 'keys {}'.format(ms_data.keys())   
    print ('evaluate scale ', ms_images.keys())
    for k in ms_images.keys():
        data = h5py.File(h5file)[k]
        images = []
        for i in range(data.shape[0]):
            img = data[i]
            # import pdb; pdb.set_trace()
            assert((img.shape[0] in [256, 128, 64, 512]) and img.shape[2] == 3)
            if not (img.min() >= 0 and img.max() <= 255 and img.mean() > 1):
                print ('WARNING {}, min {}, max {}, mean {}'.format(i, img.min(), img.max(), img.mean()))
                continue	
            #assert img.min() >= 0 and img.max() <= 255 and img.mean() > 1, '{}, min {}, max {}, mean {}'.format(i, img.min(), img.max(), img.mean())
            images.append(img)
        print ('read {} with {} images'.format(k, data.shape[0]))
        ms_images[k] = images

    print ('Totally {} images/scale are loaded at scales {}'.format(len(images), ms_images.keys() ))

    return ms_images, return_path

def load_data_from_h5_fakehr(fullpath):
    import glob
    import deepdish as dd
    import h5py
    print ('*********use fake h5 loader for hr**********')
    # import pdb; pdb.set_trace()
    h5file = os.path.join(fullpath, FLAGS.h5_file)
    return_path = os.path.join(fullpath, FLAGS.h5_file[:-3]+'_inception_score')
    print ('read h5 from {}'.format(h5file))

    
    # ms_images = {'output_64': [], 'output_128': [], 'output_256': [], }
    ms_images = {'output_512': [], }
    # assert len(ms_data.keys()) == 3, 'keys {}'.format(ms_data.keys())   

    k = 'output_256'
    print ('upsample {} to {}'.format(k, ms_images.keys()))
    assert(os.path.isfile(h5file))
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
        # ms_images['output_64'].append(img)
        # ms_images['output_128'].append(scipy.misc.imresize(img, [128, 128]))
        # ms_images['output_256'].append(scipy.misc.imresize(img, [256, 256]))
        ms_images['output_512'].append(scipy.misc.imresize(img, [512, 512]))

    print ('read {} with {} images'.format(k, data.shape[0]))

    return ms_images, return_path      


def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
    """Build Inception v3 model architecture.

    See here for reference: http://arxiv.org/abs/1512.00567

    Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

    Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
    }
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.ops.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params=batch_norm_params):
            logits, endpoints = slim.inception.inception_v3(
              images,
              dropout_keep_prob=0.8,
              num_classes=num_classes,
              is_training=for_training,
              restore_logits=restore_logits,
              scope=scope)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['aux_logits']

    return logits, auxiliary_logits


def main(unused_argv=None):
    """Evaluate model on Dataset for a number of steps."""
    
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % FLAGS.gpu):
                # Number of classes in the Dataset label set plus 1.
                # Label 0 is reserved for an (unused) background class.
                num_classes = FLAGS.num_classes + 1

                # Build a Graph that computes the logits predictions from the
                # inference model.
                inputs = tf.placeholder(
                    tf.float32, [FLAGS.batch_size, 299, 299, 3],
                    name='inputs')
                # print(inputs)

                logits, _ = inference(inputs, num_classes)
                # calculate softmax after remove 0 which reserve for BG
                known_logits = \
                    tf.slice(logits, [0, 1],
                             [FLAGS.batch_size, num_classes - 1])
                pred_op = tf.nn.softmax(known_logits)

                # Restore the moving average version of the
                # learned variables for eval.
                variable_averages = \
                    tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, FLAGS.checkpoint_dir)
                print('Restore the model from %s).' % FLAGS.checkpoint_dir)
                # images = load_data(fullpath)
                ms_images, return_save_path = load_data_from_h5(fullpath)
                # ms_images, return_save_path = load_data_from_h5_fakehr(fullpath)
                ms_means = {k:[] for k in ms_images.keys()}
                ms_std = {k:[] for k in ms_images.keys()}
                for scale, images in ms_images.items():
                    mean, std = get_inception_score(sess, images, pred_op)
                    ms_means[scale] = float(mean)
                    ms_std[scale] = float(std)
                    print ('scale: {} mean: {} std:{}'.format(scale, mean, std))
                print (ms_means, ms_std)
                json.dump({'mean': ms_means, 'std': ms_std}, open(return_save_path + '.json','w'), indent=True)

  

if __name__ == '__main__':
    tf.app.run()
