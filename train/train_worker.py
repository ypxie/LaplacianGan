# -*- coding: utf-8 -*-

import numpy as np
import argparse, os
import torch, h5py

import torch.nn as nn

from LaplacianGan.paraGan import train_gans
from LaplacianGan.fuel.zz_datasets import TextDataset

def train_worker(data_root, model_root, training_dict):

    save_freq           = getattr(training_dict, 'save_freq', 3)
    ncritic_epoch_range = getattr(training_dict, 'ncritic_epoch_range', 600)
    g_lr                = getattr(training_dict, 'g_lr', .0002)
    d_lr                = getattr(training_dict, 'd_lr', .0002)

    parser = argparse.ArgumentParser(description = 'Gans')    
    parser.add_argument('--weight_decay', type=float, default= 0,
                        help='weight decay for training')
    parser.add_argument('--maxepoch', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default = g_lr, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default = d_lr, metavar='LR',
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--reuse_weights',  default= training_dict['reuse_weights'],
                        help='continue from last checkout point')
    parser.add_argument('--show_progress', action='store_false', default = True,
                        help='show the training process using images')
    
    parser.add_argument('--save_freq', type=int, default= save_freq, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default= 200, metavar='N',
                        help='plot the results every {} batches')
    parser.add_argument('--verbose_per_iter', type=int, default= 50, 
                        help='print losses per iteration')
    parser.add_argument('--batch_size', type=int, default=training_dict['batch_size'], metavar='N',
                        help='batch size.')
    parser.add_argument('--num_emb', type=int, default=4, metavar='N',
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
    parser.add_argument('--use_content_loss', type=bool, default= True, metavar='N',
                        help='whether or not to use content loss.')

    ## add more
    parser.add_argument('--device_id', type=int, default=training_dict['device_id'], 
                        help='which device')
    parser.add_argument('--gpu_list',  default=training_dict['gpu_list'], 
                        help='which devices to parallel the data')
    parser.add_argument('--imsize', type=int, default=training_dict['imsize'], 
                        help='output image size')
    parser.add_argument('--epoch_decay', type=float, default=100, 
                        help='decay epoch image size')
    parser.add_argument('--load_from_epoch', type=int, default= training_dict['load_from_epoch'], 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default=training_dict['model_name'])
    parser.add_argument('--test_sample_num', type=int, default=4, 
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--norm_type', type=str, default='bn', 
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--gen_activation_type', type=str, default='relu', 
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--debug_mode', type=bool, default=True,  
                        help='debug mode use fake dataset loader')   
    parser.add_argument('--which_gen', type=str, default=training_dict['which_gen'],  help='generator type')
    parser.add_argument('--which_disc', type=str, default=training_dict['which_disc'], help='discriminator type')
    
    parser.add_argument('--dataset', type=str, default=training_dict['dataset'], help='which dataset to use [birds or flowers]') 
    parser.add_argument('--ncritic_epoch_range', type=int, default=ncritic_epoch_range, help='How many epochs the ncritic effective')   

    args = parser.parse_args()

    args.cuda  = torch.cuda.is_available()
    
    data_name  = args.dataset
    datadir = os.path.join(data_root, data_name)

    # Generator
    if args.which_gen == 'origin':
        from LaplacianGan.models.hd_networks import Generator
        netG = Generator(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, 
                        norm=args.norm_type, activation=args.gen_activation_type, output_size=args.imsize)
    elif args.which_gen == 'upsample_skip':   
        from LaplacianGan.models.hd_networks import Generator 
        netG = Generator(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, 
                        norm=args.norm_type, activation=args.gen_activation_type, output_size=args.imsize, use_upsamle_skip=True)              
    else:
        raise NotImplementedError('Generator [%s] is not implemented' % args.which_gen)

    # Discriminator
    if args.which_disc == 'origin': 
        # only has global discriminator
        from LaplacianGan.models.hd_networks import Discriminator 
        netD = Discriminator(input_size=args.imsize, num_chan = 3, hid_dim = 128, 
                    sent_dim=1024, emb_dim=128, norm=args.norm_type)
    elif args.which_disc == 'origin_global_local':
        # has global and local discriminator
        from LaplacianGan.models.hd_networks import Discriminator 
        netD = Discriminator(input_size=args.imsize, num_chan = 3, hid_dim = 128, 
                    sent_dim=1024, emb_dim=128, norm=args.norm_type, disc_mode=['global', 'local'])
    else:
        raise NotImplementedError('Discriminator [%s] is not implemented' % args.which_disc)
    
    #print(netG)
    #print(netD) 
    print(args)

    device_id = getattr(args, 'device_id', 0)

    if args.cuda:
        netD = netD.cuda(device_id)
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    
    netG = nn.DataParallel(netG, device_ids=args.gpu_list, output_device= args.device_id)
    netD = nn.DataParallel(netD, device_ids=args.gpu_list, output_device= args.device_id)

    if not args.debug_mode:
        print ('>> initialize dataset')
        dataset = TextDataset(datadir, 'cnn-rnn', 4)
        filename_test = os.path.join(datadir, 'test')
        dataset.test = dataset.get_data(filename_test)
        filename_train = os.path.join(datadir, 'train')
        dataset.train = dataset.get_data(filename_train)
    else:
        dataset = []
        print ('>> in debug mode')
    model_name ='{}_{}_{}'.format(args.model_name, data_name, args.imsize)
    print ('>> START training ')
    train_gans(dataset, model_root, model_name, netG, netD, args)
