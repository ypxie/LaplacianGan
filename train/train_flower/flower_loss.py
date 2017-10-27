import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..','..'))

data_root  = os.path.join('..','..', 'Data')
model_root = os.path.join( '..','..', 'Models')

import torch.multiprocessing as mp

from LaplacianGan.proj_utils.local_utils import Indexflow
from LaplacianGan.train_worker import train_worker

# local_global disc. We test both large and small model
# 101 201 301 401 501 601
# 2   4   8   16  32  64

# large_global_no_img   =  {'reuse_weights': False, 'batch_size': 16, 'device_id': 3,  
#                           'g_lr': .0002/(2**0),'d_lr': .0002/(2**0), 
#                           'imsize':[64, 128, 256], 'load_from_epoch': 0, 'model_name':'gen_origin_disc_global_no_img',
#                           'use_img_loss' : False, 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'flowers',
#                           'reduce_dim_at':[8, 32, 128, 256] }

# large_both            = {'reuse_weights': True, 'batch_size': 16, 'device_id': 0,  
#                          'g_lr': .0002/(2**5),'d_lr': .0002/(2**5),
#                          'imsize':[64, 128, 256], 'load_from_epoch': 504, 'model_name':'gen_origin_disc_both', 
#                          'which_gen': 'origin', 'which_disc':'origin_global_local', 'dataset':'flowers',
#                          'reduce_dim_at':[8, 32, 128, 256]}

large_local          = { 'reuse_weights': True, 'batch_size': 16, 'device_id': 1,  
                         'g_lr': .0002/(3**3),'d_lr': .0002/(3**3), 'img_loss_ratio': 1, 'tune_img_loss': False,
                         'imsize':[64, 128, 256], 'load_from_epoch': 360, 'model_name':'gen_origin_disc_local', 
                         'which_gen': 'origin', 'which_disc':'local', 'dataset':'flowers',
                         'reduce_dim_at':[8, 32, 128, 256]}

large_local_low      = {'reuse_weights': False, 'batch_size': 16, 'device_id': 0,  
                         'g_lr': .0002/(3**0),'d_lr': .0002/(3**0), 'img_loss_ratio': 0.5, 'tune_img_loss':True,
                         'imsize':[64, 128, 256], 'load_from_epoch': 0, 'model_name':'gen_origin_disc_local_low', 
                         'which_gen': 'origin', 'which_disc':'local', 'dataset':'flowers',
                         'reduce_dim_at':[8, 32, 128, 256]}



training_pool = np.array([
                 #large_global_no_img,
                 large_local,
                 large_local_low
                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

for select_ind in Indexflow(Totalnum, 2, random=False):
    select_pool = training_pool[select_ind]
    print('selcted training pool: ', select_pool)
    
    for this_dick in select_pool:

        p = mp.Process(target=train_worker, args= (data_root, model_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with', select_pool)

