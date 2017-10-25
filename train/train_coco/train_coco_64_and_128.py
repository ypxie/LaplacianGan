import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..','..'))

data_root  = os.path.join('..','..', 'Data')
model_root = os.path.join( '..','..', 'Models')

import torch.multiprocessing as mp

from LaplacianGan.proj_utils.local_utils import Indexflow
from LaplacianGan.train_worker_coco import train_worker

reduce_dim_at = [8, 32, 128, 256]
large_global = {'reuse_weights': False, 'batch_size': 22, 'device_id': 6, 'gpu_list': [0], 
                'imsize':[64, 128], 'load_from_epoch': 36, 'model_name':'gen_origin_disc_origin', 
                'which_gen': 'origin', 'which_disc':'origin', 'dataset':'coco',
                'reduce_dim_at':[8, 32, 128, 256], 'num_resblock':2 }

small_global = {'reuse_weights': False, 'batch_size': 64, 'device_id': 7, 'gpu_list': [0], 
                'imsize':[64], 'load_from_epoch': 48, 'model_name':'gen_origin_disc_origin', 
                'which_gen': 'origin', 'which_disc':'origin', 'dataset':'coco',
                'reduce_dim_at':[8, 32, 128, 256], 'num_resblock':2 }

training_pool = np.array([
                 large_global,
                 small_global
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

