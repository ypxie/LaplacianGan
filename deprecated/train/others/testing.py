# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.join('..','..'))

import numpy as np
import argparse, os
import torch, h5py
from torch.nn.utils import weight_norm

from LaplacianGan.models.refineModels import Discriminator as Disc
from LaplacianGan.models.refineModels import Generator as Gen
from LaplacianGan.lapGan import train_gans
#from LaplacianGan.zzGan import train_gans

from LaplacianGan.fuel.datasets import TextDataset


home = os.path.expanduser('~')
data_root = os.path.join('..', '..', 'Data')

model_root = os.path.join('..', '..', 'Models')
data_name = 'birds'
datadir = os.path.join(data_root, data_name)


device_id = 2

if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gans')    
    parser.add_argument('--weight_decay', type=float, default= 0,
                        help='weight decay for training')
    parser.add_argument('--maxepoch', type=int, default=12800000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default = .0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default = .0002, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--reuse_weights', action='store_false', default = True,
                        help='continue from last checkout point')
    parser.add_argument('--show_progress', action='store_false', default = True,
                        help='show the training process using images')
    
    parser.add_argument('--save_freq', type=int, default= 500, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default= 200, metavar='N',
                        help='plot the results every {} batches')
    
    parser.add_argument('--batch_size', type=int, default= 16, metavar='N',
                        help='batch size.')
    parser.add_argument('--num_emb', type=int, default= 4, metavar='N',
                        help='number of emb chosen for each image.')

    parser.add_argument('--gp_lambda', type=int, default=10, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--wgan', action='store_false', default=  False,
                        help='enables gradient penalty')
    
    parser.add_argument('--noise_dim', type=int, default= 100, metavar='N',
                        help='dimension of gaussian noise.')
    parser.add_argument('--ncritic', type=int, default= 1, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--ngen', type=int, default= 1, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--KL_COE', type=float, default= 4, metavar='N',
                        help='kl divergency coefficient.')
    parser.add_argument('--use_content_loss', type=bool, default= False, metavar='N',
                        help='whether or not to use content loss.')
    parser.add_argument('--save_folder', type=str, default= 'tmp_images', metavar='N',
                        help='folder to save the temper images.')
    
    ## add more
    parser.add_argument('--imsize', type=int, default=256, 
                        help='output image size')
    parser.add_argument('--epoch_decay', type=float, default=100, 
                        help='decay epoch image size')
    parser.add_argument('--load_from_epoch', type=int, default=0, 
                        help='load from epoch')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    img_size, lratio = 256, 4
    norm = 'bn'

    dataset = TextDataset(datadir, 'cnn-rnn', lratio)
    filename_test = os.path.join(datadir, 'test')
    dataset.test = dataset.get_data(filename_test)
    filename_train = os.path.join(datadir, 'train')
    dataset.train = dataset.get_data(filename_train)
    train_sampler = dataset.train.next_batch

    train_sampler(4, 4)