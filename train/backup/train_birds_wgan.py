# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.join('..'))

import numpy as np
import argparse, os
import torch, h5py
from torch.nn.utils import weight_norm

from LaplacianGan.models.stackModels import Discriminator as Disc
from LaplacianGan.models.stackModels import Generator as Gen
from LaplacianGan.lapGan import train_gans

from LaplacianGan.fuel.datasets import TextDataset


home = os.path.expanduser('~')
dropbox = os.path.join(home, 'Dropbox')
data_root = os.path.join('..', 'Data')

data_name = 'birds'
datadir = os.path.join(data_root, data_name)


device_id = 0

if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gans')    
    parser.add_argument('--weight_decay', type=float, default= 0,
                        help='weight decay for training')
    parser.add_argument('--maxepoch', type=int, default=12800000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default = 0.00005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default = 0.00005, metavar='LR',
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
    
    parser.add_argument('--batch_size', type=int, default= 64, metavar='N',
                        help='batch size.')
    parser.add_argument('--num_emb', type=int, default= 4, metavar='N',
                        help='number of emb chosen for each image.')

    parser.add_argument('--gp_lambda', type=int, default=10, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--wgan', action='store_false', default=  True,
                        help='enables gradient penalty')
    
    parser.add_argument('--noise_dim', type=int, default= 100, metavar='N',
                        help='dimension of gaussian noise.')
    parser.add_argument('--ncritic', type=int, default= 5, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--ngen', type=int, default= 1, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--KL_COE', type=float, default= 4, metavar='N',
                        help='kl divergency coefficient.')
    parser.add_argument('--use_content_loss', type=bool, default= True, metavar='N',
                        help='whether or not to use content loss.')

    parser.add_argument('--save_folder', type=str, default= 'tmp_images', metavar='N',
                        help='folder to save the temper images.')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    img_size, lratio = 64, 1
    
    netG = Gen(input_size  = img_size, sent_dim= 1024, noise_dim = args.noise_dim, 
               num_chan=3, emb_dim= 128, hid_dim= 128, norm='bn', branch=False)

    netD = Disc(input_size = img_size, num_chan = 3, hid_dim = 64, 
                sent_dim=1024, enc_dim = 256, emb_dim= 128,  norm='bn')
                            
    if args.cuda:
        netD = netD.cuda(device_id)
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print(netG)
    print(netD)
    
    dataset = TextDataset(datadir, 'cnn-rnn', lratio)
    filename_test = os.path.join(datadir, 'test')
    dataset.test = dataset.get_data(filename_test)

    filename_train = os.path.join(datadir, 'train')
    dataset.train = dataset.get_data(filename_train)

    model_root, model_name = 'model', 'lap_wgan_{}_{}'.format(data_name, img_size)
    train_gans(dataset, model_root, model_name, netG, netD,args)
