# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.join('..'))

import numpy as np
import argparse, os
import torch, h5py
from torch.nn.utils import weight_norm

from LaplacianGan.HDGan import train_gans
from LaplacianGan.fuel.zz_datasets import TextDataset
from LaplacianGan.testGan import test_gans, generate_layer_features
from LaplacianGan.proj_utils.local_utils import mkdirs

home = os.path.expanduser('~')
proj_root = os.path.join('..')
data_root = os.path.join(proj_root, 'Data')

data_name = 'birds'
datadir = os.path.join(data_root, data_name)
model_root = os.path.join('..', 'Models')
# model_root = os.path.join(home, 'devbox', 'Shared_YZ', 'models')


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
    parser.add_argument('--save_spec', type=str, default='', help='save_spec')
    parser.add_argument('--train_mode',action='store_true', 
                        help='continue from last checkout point')
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
    elif args.which_gen == 'single_256':   
        from LaplacianGan.models.hd_networks import Generator 
        netG = Generator(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, 
                        norm=args.norm_type, activation=args.gen_activation_type, output_size=[256])   
    elif args.which_gen == 'comb_64_256':   
        from LaplacianGan.models.hd_networks import Generator 
        netG = Generator(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, 
                        norm=args.norm_type, activation=args.gen_activation_type, output_size=[64, 256])           
    elif args.which_gen == 'super':   
        from LaplacianGan.models.hd_networks import GeneratorSuper
        netG = GeneratorSuper(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, 
            norm=args.norm_type, activation=args.gen_activation_type)   
    elif args.which_gen == 'super_small':   
        from LaplacianGan.models.hd_networks import GeneratorSuperSmall
        netG = GeneratorSuperSmall(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, 
            norm=args.norm_type, activation=args.gen_activation_type) 
        print(netG) 
    else:
        raise NotImplementedError('Generator [%s] is not implemented' % args.which_gen)

    device_id = getattr(args, 'device_id', 0)

    if args.cuda:
        # netD = netD.cuda(device_id)
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print ('>> initialize dataset')
    dataset = TextDataset(datadir, 'cnn-rnn', 4)
    filename_test = os.path.join(datadir, 'test')
    dataset.test = dataset.get_data(filename_test)

    model_name ='{}_{}_{}'.format(args.model_name, data_name, args.imsize)

    # save_root = os.path.join(home, 'devbox', 'Shared_YZ', 'Results',data_name, args.save_spec + 'testing_num_{}'.format(args.test_sample_num) )
   
    save_root = os.path.join(model_root, model_name, 'results/')
    if not os.path.isdir(save_root):
        mkdirs(save_root)

    # test_gans(dataset, model_root, model_name, save_root, netG, args)
    generate_layer_features(dataset, model_root, model_name, save_root, netG, args)
    
    