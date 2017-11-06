import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..','..'))

home = os.path.expanduser("~")
#data_root  = os.path.join('..','..', 'Data')
#model_root = os.path.join( '..','..', 'Models')

data_root  = os.path.join(home, 'ganData')
model_root = os.path.join(data_root, 'Models')

import torch.multiprocessing as mp
from LaplacianGan.proj_utils.local_utils import Indexflow
from LaplacianGan.train_worker_coco import train_worker

reduce_dim_at = [8, 32, 128, 256]
# coco_256 = {'reuse_weights': True, 'batch_size': 12, 'device_id': 0, 'gpu_list': [0], 
#             'img_loss_ratio':0.5/(2**3), 'tune_img_loss':True, 'g_lr': .0002/(2**3),  'd_lr': .0002/(2**3), 
#             'imsize':[64, 128, 256], 'load_from_epoch': 60, 'model_name':'gen_origin_disc_local', 
#             'which_gen': 'origin', 'which_disc':'local', 'dataset':'coco',
#             'reduce_dim_at':[8, 32, 128, 256], 'num_resblock':2 }

# coco_256_fine = {'reuse_weights': True, 'batch_size': 12, 'device_id': 1, 'gpu_list': [0],'img_loss_ratio':0.2,
#                  'detach_list':[64], 'load_from_epoch': 9, 
#                  'imsize':[64, 128, 256], 'model_name':'gen_origin_disc_local_finetune', 
#                  'which_gen': 'origin', 'which_disc':'local', 'dataset':'coco',
#                  'reduce_dim_at':[8, 32, 128, 256], 'num_resblock':2 }

# coco_128 = {'reuse_weights': False, 'batch_size': 22, 'device_id': 2, 'gpu_list': [0], 
#             'imsize':[64, 128], 'load_from_epoch': 84, 'model_name':'gen_origin_disc_local', 
#             'which_gen': 'origin', 'which_disc':'local', 'dataset':'coco',
#             'reduce_dim_at':[8, 32, 128, 256], 'num_resblock':2 }

# coco_64 = {'reuse_weights':  True, 'batch_size': 64, 'device_id': 0, 'gpu_list': [0], 
#             'imsize':[64], 'load_from_epoch': 273, 'model_name':'gen_origin_disc_local', 
#             'g_lr': .0002/(2**3),  'd_lr': .0002/(2**3),  'save_freq': 10,
#             'which_gen': 'origin', 'which_disc':'local', 'dataset':'coco',
#             'reduce_dim_at':[8, 32, 128, 256], 'num_resblock':2 }

testing_inplace_kl_64 = {'reuse_weights':  False, 'batch_size': 64, 'device_id': 0, 'gpu_list': [0], 
                        'imsize':[64], 'load_from_epoch': 0, 'model_name':'just_fortesting', 
                        'g_lr': .0002/(2**0),  'd_lr': .0002/(2**0),  'save_freq': 10,
                        'which_gen': 'origin', 'which_disc':'local', 'dataset':'coco',
                        'reduce_dim_at':[8, 32, 128, 256], 'num_resblock':2 }

# coco_64_256 = {         'reuse_weights':  False, 'batch_size': 12, 'device_id': 0, 'gpu_list': [0], 
#                         'imsize':[64, 256], 'load_from_epoch': 273, 'model_name':'gen_origin_disc_local', 
#                         'g_lr': .0002/(2**0),  'd_lr': .0002/(2**0),  'save_freq': 3,
#                         'which_gen': 'origin', 'which_disc':'local', 'dataset':'coco',
#                         'reduce_dim_at':[8, 32, 128, 256], 'num_resblock':2 }


training_pool = np.array([
                 testing_inplace_kl_64,
                 #  coco_64_256
                 # coco_256,
                 # coco_256_fine
                 # coco_128,
                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

for select_ind in Indexflow(Totalnum, 2, random=False):
    select_pool = training_pool[select_ind]
    print('selcted training pool: ', select_ind)
    
    for this_dick in select_pool:

        p = mp.Process(target=train_worker, args= (data_root, model_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with', select_ind)

