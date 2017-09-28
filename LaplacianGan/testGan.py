import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter

from torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
from .proj_utils.torch_utils import *
from .zzGan import load_partial_state_dict

import time, json

TINY = 1e-8

def test_gans(dataset, model_root, mode_name, save_root , netG,  args):
    # helper function

    test_sampler  = dataset.test.next_batch_test

    model_folder = os.path.join(model_root, mode_name)
    model_marker = mode_name + '_epoch_{}'.format(args.load_from_epoch)

    save_folder  = os.path.join(save_root, model_marker )   # to be defined in the later part
    save_h5 = os.path.join(save_root, model_marker+'.h5')

    mkdirs(save_folder)
    
    ''' load model '''
    assert args.load_from_epoch != '', 'args.load_from_epoch is empty'
    G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
    print('reload weights from {}'.format(G_weightspath))
    weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
    load_partial_state_dict(netG, weights_dict)

    # test the fixed image for every epoch
    fixed_images, fixed_embeddings, _, _ = test_sampler(args.batch_size, 1)
    fixed_embeddings = to_device(fixed_embeddings, netG.device_id, volatile=True)
    fixed_z_data = [torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1) for _ in range(args.test_sample_num)]
    fixed_z_list = [to_device(a, netG.device_id, volatile=True) for a in fixed_z_data]

    testing_z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    testing_z = to_device(testing_z, netG.device_id, volatile=True)    
    

    while True:
        if dataset.test.end_of_data:
            break;
        
        test_images, test_embeddings_list, saveIDs, test_captions = test_sampler(args.batch_size, 1)
        #test_embeddings_list is a list of (B,emb_dim)
        ''' visualize test per epoch '''
        # generate samples
        gen_samples = []
        img_samples = []
        vis_samples = {}

        tmp_samples = {}
        
        data_dict = {}
        all_captions = []
        for t in range(args.test_sample_num):
            
            B = len(test_embeddings_list)
            ridx = random.randint(0, B-1)
            testing_z.data.normal_(0, 1)

            this_test_embeddings = test_embeddings_list[ridx]
            samples, _ = netG(test_embeddings, testing_z)
            
            if  t == 0:  
                for k in samples.keys():
                    vis_samples[k] = [None for i in range(args.test_sample_num + 1)] # +1 to fill real image
                    data_dict[k] = []

            for k, v in samples.items():
                cpu_data = v.cpu().data.numpy()
                if t == 0:
                    if vis_samples[k][0] == None:
                        vis_samples[k][0] = test_images[k]
                    else:
                        vis_samples[k][0] =  np.concatenate([ vis_samples[k][0], test_images[k]], 0) 

                if vis_samples[k][t+1] == None:
                    vis_samples[k][t+1] = cpu_data
                else:
                    vis_samples[k][t+1] = np.concatenate([vis_samples[k][t+1], cpu_data], 0)

        for typ, v in vis_samples.items():

            data_dict[typ].append( np.stack(v, 1)  ) # list of N*T*3*row*col

            for idx, img_id in enumerate(saveIDs):
                this_img  = v[idx:idx+1]
                save_path = os.path.join(save_folder, '{}_{}.png'.format(img_id, typ) )
                save_images(this_img, save = True, save_path=save_path, dim_ordering='th')
    
    string_dt = h5py.special_dtype(vlen=str)
    #dset_img = h5file.create_dataset("images", (total_num, 3, 224, 224), dtype='uint8')
            
    with h5py.File(save_h5,'w') as h5file:
        for k, v in data_dict.items():
            all_data = np.concatenate(v, 0)
            dset_img = h5file.create_dataset(k, data=all_data, dtype='float32')
            
            h5file.create_dataset("captions", data=np.array(test_captions), dtype=string_dt)
            h5file.close()