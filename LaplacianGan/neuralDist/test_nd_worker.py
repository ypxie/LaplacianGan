# -*- coding: utf-8 -*-

import numpy as np
import argparse, os
import torch, h5py

import torch.nn as nn
from collections import OrderedDict
from .proj_utils.local_utils import mkdirs
from .trainNeuralDist  import train_nd
from .neuralDistModel  import ImgSenRanking
from .neuralDistModel  import ImageEncoder

def test_worker(data_root, model_root, save_root, testing_dict):
    print('testing_dict: ', testing_dict)

    batch_size   =  testing_dict.get('batch_size')
    device_id    =  testing_dict.get('device_id')
    
    dim_image    =  testing_dict.get('dim_image', 1536) 
    sent_dim     =  testing_dict.get('sent_dim',  1024) 
    hid_dim      =  testing_dict.get('hid_dim',    512) 

    parser = argparse.ArgumentParser(description = 'test nd') 
    parser.add_argument('--batch_size', type=int, default=batch_size, metavar='N',
                        help='batch size.')
    
    parser.add_argument('--device_id', type=int, default= device_id, 
                        help='which device')
    
    parser.add_argument('--load_from_epoch', type=int, default= testing_dict['load_from_epoch'], 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default = testing_dict['model_name'])

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    
    
    data_name  = testing_dict['dataset']
    datadir    = os.path.join(data_root, data_name)

    device_id = getattr(args, 'device_id', 0)
    print('device_id: ', device_id)
    if args.cuda:
        #netD = netD.cuda(device_id)
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print ('>> initialize dataset')
    dataset = TextDataset(datadir, 'cnn-rnn', 4)
    filename_test = os.path.join(datadir, 'test')
    dataset.test = dataset.get_data(filename_test)

    model_name = args.model_name   #'{}_{}_{}'.format(args.model_name, data_name, args.imsize)


    save_folder  = os.path.join(save_root, data_name, args.save_spec + 'testing_num_{}'.format(args.test_sample_num) )
    mkdirs(save_folder)

    test_gans(dataset, model_root, model_name, save_folder, netG, args)

    
    