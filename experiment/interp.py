import os, pickle
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..'))

home = os.path.expanduser("~")
data_root  = os.path.join('..', 'Data')
model_root = os.path.join('..', 'Models')

#data_root  = os.path.join(home, 'ganData')
#model_root = os.path.join(data_root, 'Models')
#model_root = os.path.join(home, 'devbox', 'Shared_YZ', 'models')

save_root  =  os.path.join(home, 'devbox', 'Shared_YZ', 'Results')


import numpy as np
import os, argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter

from torch.nn.utils import clip_grad_norm
from LaplacianGan.proj_utils.plot_utils import *
from LaplacianGan.proj_utils.local_utils import *

from LaplacianGan.proj_utils.torch_utils import *
from LaplacianGan.zzGan import load_partial_state_dict
from LaplacianGan.testGan import save_super_images

from PIL import Image, ImageDraw, ImageFont

import time, json, h5py
import deepdish as dd

TINY = 1e-8


def test_gans(dict_left, dict_right, testing_z, epsilon,
              model_root, mode_name, save_root, netG, args):
    # helper function
    if args.train_mode:
        print('Using training mode')
        netG.train()
    else:
        print('Using testing mode')
        netG.eval()

    #test_sampler  = dataset.test.next_batch_test

    model_folder = os.path.join(model_root, mode_name)
    model_marker = mode_name + '_G_epoch_{}'.format(args.load_from_epoch)

    save_folder  = os.path.join(save_root, model_marker )   # to be defined in the later part
    save_h5    = os.path.join(save_root, model_marker+'.h5')
    org_h5path = os.path.join(save_root, 'original.h5')
    mkdirs(save_folder)
    
    ''' load model '''
    assert args.load_from_epoch != '', 'args.load_from_epoch is empty'
    G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
    print('reload weights from {}'.format(G_weightspath))
    weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
    load_partial_state_dict(netG, weights_dict)
    
    
    
    testing_z = to_device(testing_z, netG.device_id, volatile=True)
    epsilon   = to_device(epsilon, netG.device_id, volatile=True)
    num_examples = 1 #dataset.test._num_examples
    #dataset.test._num_examples = num_examples
    test_sample_num = len(args.alpha_list)
    total_number = num_examples * test_sample_num
    
    all_choosen_caption = []
    org_file_not_exists = not os.path.exists(org_h5path)
    
    if org_file_not_exists:
        org_h5 = h5py.File(org_h5path,'w')
        org_dset = org_h5.create_dataset('output_256', shape=(num_examples,256, 256,3), dtype=np.uint8)
    else:
        org_dset = None
    with h5py.File(save_h5,'w') as h5file:

        start_count = 0
        data_count = {}
        dset = {}
        gen_samples = []
        img_samples = []
        vis_samples = {}
        tmp_samples = {}
        init_flag = True
        
        test_emb_1, test_captions_1 = dict_left['emb'],   dict_left['caption']
        test_emb_2, test_captions_2 = dict_right['emb'],  dict_right['caption']
        
        chosen_captions =  test_captions_1 + test_captions_2
        this_batch_size =  test_emb_1.shape[0]
        saveIDs         =  [1]*this_batch_size

        #print('start: {}, this_batch size {}, num_examples {}'.format(start_count, test_images.shape[0], dataset.test._num_examples  ))
        # chosen_captions = []
        # for this_caption_list in test_captions:
        #     chosen_captions.append(this_caption_list[0])
        #all_choosen_caption.extend(chosen_captions)    

        start_count += this_batch_size
        
        #test_embeddings_list is a list of (B,emb_dim)
        ''' visualize test per epoch '''
        # generate samples
        
        for test_idx, t in enumerate(range(test_sample_num) ):
            
            # B = len(test_embeddings_list)
            # ridx = random.randint(0, B-1)
            # testing_z.data.normal_(0, 1)
            #this_test_embeddings = test_embeddings_list[ridx]
            this_alpha = args.alpha_list[test_idx]
            this_test_embeddings  = this_alpha*test_emb_1 + (1-this_alpha)*test_emb_2
            this_test_embeddings  = to_device(this_test_embeddings, netG.device_id, volatile=True)
            test_outputs, _ = netG(this_test_embeddings, testing_z[0:this_batch_size], epsilon)
            
            if  t == 0: 
                if init_flag is True:
                    dset['captions'] = h5file.create_dataset('captions', shape=(total_number,), dtype=np.int64)

                    for k in test_outputs.keys():
                        vis_samples[k] = [None for i in range(test_sample_num)] # +1 to fill real image
                        img_shape = test_outputs[k].size()[2::]

                        print('total number of images is: ', total_number)
                        dset[k] = h5file.create_dataset(k, shape=(total_number,) + img_shape + (3,), dtype=np.uint8)
                        data_count[k] = 0
                init_flag = False
            
            for typ, img_val in test_outputs.items():
                cpu_data = img_val.cpu().data.numpy()
                #print('cpu_data shape: ', cpu_data.shape)
                vis_samples[typ][t] = cpu_data

                bs = cpu_data.shape[0]
                
                start = data_count[typ]
                this_sample = ((cpu_data + 1) * 127.5 ).astype(np.uint8)
                this_sample = this_sample.transpose(0, 2,3,1)

                dset[typ][start: start + bs] = this_sample
                #dset['classIDs'][start: start + bs] = classIDs
                data_count[typ] = start + bs
                #print('saved files: ', data_count) 

        if args.save_images:
            save_super_images(vis_samples, chosen_captions, this_batch_size, save_folder, saveIDs)
             
        #caption_array = np.array(all_choosen_caption, dtype=object)
        #string_dt = h5py.special_dtype(vlen=str)
        #h5file.create_dataset("captions", data=caption_array, dtype=string_dt)
        if org_dset is not None:
            org_h5.close() 


def get_args(testing_dict):
    parser = argparse.ArgumentParser(description = 'Gans')    

    parser.add_argument('--noise_dim', type=int, default= 100, metavar='N',
                        help='dimension of gaussian noise.')

    parser.add_argument('--num_emb', type=int, default=4, metavar='N',
                        help='number of emb chosen for each image.')
    ## add more
    parser.add_argument('--device_id', type=int, default= testing_dict['device_id'], 
                        help='which device')
    parser.add_argument('--imsize', type=int, default= testing_dict['imsize'], 
                        help='output image size')
    parser.add_argument('--epoch_decay', type=float, default=100, 
                        help='decay epoch image size')
    parser.add_argument('--load_from_epoch', type=int, default= testing_dict['load_from_epoch'], 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default = testing_dict['model_name'])

    
    parser.add_argument('--norm_type', type=str, default='bn', 
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--gen_activation_type', type=str, default='relu', 
                        help='The number of runs for each embeddings when testing')
                        
    parser.add_argument('--which_gen', type=str, default=testing_dict['which_gen'],  help='generator type')

    parser.add_argument('--save_spec', type=str, default = testing_dict['save_spec'], help='save_spec')
    parser.add_argument('--train_mode', type=bool, default = testing_dict['train_mode'],
                        help='continue from last checkout point')
    parser.add_argument('--save_images', type=bool, default = True,
                        help='do you really want to save big images to folder')
    parser.add_argument('--reduce_dim_at', default= testing_dict['reduce_dim_at'], help='Reduce the dimension at.')
    parser.add_argument('--alpha_list', default= None, help='The list of alpha values')

    parser.add_argument('--dataset', default= testing_dict['dataset'], help='dataset')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    return args

if __name__ == "__main__":
    
    save_spec = 'interpolation'
    gen_origin_disc_global_local_low_birds_597  =   \
               { 'test_sample_num' : 10,  'load_from_epoch': 327, 'dataset':'birds', 
                 'device_id': 3, 'imsize':256, 'model_name':'gen_origin_disc_local_low_birds_[64, 128, 256]',
                 'train_mode': False,  'save_spec': save_spec,  'which_gen': 'origin',
                 'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }
    
    args =  get_args(gen_origin_disc_global_local_low_birds_597)
    
    data_name  = args.dataset
    #datadir    = os.path.join(data_root, data_name)

    # Generator
    from LaplacianGan.models.hd_bugfree import Generator
    netG = Generator(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, 
                     norm=args.norm_type, activation=args.gen_activation_type, 
                     output_size=args.imsize, reduce_dim_at = args.reduce_dim_at)

    torch.manual_seed(12345)
    np.random.seed(12345)
    # make it on cuda device.
    if args.cuda:
        netG = netG.cuda(args.device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    interv = 1.0/5
    args.alpha_list =  np.linspace(0,1,11)
    test_sample_num =  len(args.alpha_list)
    path_left  =  None
    path_right =  None
    with open(path_left, 'rb') as f:
        dict_left_list   = pickle.load(f)
    with open(path_right, 'rb') as f:
        dict_right_right   = pickle.load(f)    
    
    for dict_left, dict_right in zip(dict_left_list, dict_right_right)
        dict_left  = {'emb':np.random.rand(1, 1024), 'caption':'random captions'}
        dict_right = {'emb':np.random.rand(1, 1024), 'caption':'random captions'}
        
        args.batch_size = dict_left['emb'].shape[0]
        testing_z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
        epsilon   = torch.FloatTensor(args.batch_size, 128).normal_(0, 1)
        
        print({'dict_left':dict_left, 'z':testing_z , 'eps':epsilon})

        save_folder  = os.path.join(save_root, data_name, args.save_spec + 'testing_num_{}'.format(test_sample_num) )
        
        mkdirs(save_folder)
        
        test_gans(dict_left, dict_right, testing_z, epsilon, model_root, args.model_name, save_folder, netG, args)
        
