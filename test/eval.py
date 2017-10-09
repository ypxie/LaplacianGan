# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.join('..'))

import numpy as np
import argparse, os
import torch, h5py
from torch.nn.utils import weight_norm

from LaplacianGan.HDGan import train_gans
from LaplacianGan.fuel.zz_datasets import TextDataset
from LaplacianGan.testGan import test_gans
from LaplacianGan.proj_utils.local_utils import mkdirs

home = os.path.expanduser('~')
proj_root = os.path.join('..')
data_root = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')
data_name = 'birds'
datadir = os.path.join(data_root, data_name)
save_root = os.path.join(data_root, 'Results', data_name)
mkdirs(save_root)
save_h5 = os.path.join(save_root, 'zz_mmgan_plain_gl_disc_birds_256_G_epoch_500.h5')

with h5py.File(save_h5,'r') as h5file:
    img_256 = h5file['output_256']
    print(np.mean(img_256))