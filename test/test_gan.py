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

if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gans')    

    parser.add_argument('--noise_dim', type=int, default= 100, metavar='N',
                        help='dimension of gaussian noise.')

    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='batch size.')
    parser.add_argument('--num_emb', type=int, default=4, metavar='N',
                        help='number of emb chosen for each image.')
    ## add more
    parser.add_argument('--device_id', type=int, default=0, 
                        help='which device')
    parser.add_argument('--imsize', type=int, default=256, 
                        help='output image size')
    parser.add_argument('--epoch_decay', type=float, default=100, 
                        help='decay epoch image size')
    parser.add_argument('--load_from_epoch', type=int, default= 0, 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default='zz_gan')
    parser.add_argument('--test_sample_num', type=int, default= 10, 
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--norm_type', type=str, default='bn', 
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--gen_activation_type', type=str, default='relu', 
                        help='The number of runs for each embeddings when testing')
                        
    parser.add_argument('--which_gen', type=str, default='origin',  help='generator type')
    parser.add_argument('--which_disc', type=str, default='origin', help='discriminator type')
    parser.add_argument('--emb_interp', action='store_true', 
                        help='Use interpolation emb in disc')        
            
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    
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
    
    print(netG)
    print(netD) 
    
    device_id = getattr(args, 'device_id', 0)

    if args.cuda:
        netD = netD.cuda(device_id)
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print ('>> initialize dataset')
    dataset = TextDataset(datadir, 'cnn-rnn', 4)
    filename_test = os.path.join(datadir, 'test')
    dataset.test = dataset.get_data(filename_test)

    model_name ='{}_{}_{}'.format(args.model_name, data_name, args.imsize)

    test_gans(dataset, model_root, model_name, save_root, netG, args)

    
    