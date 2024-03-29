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
from LaplacianGan.train_worker import train_worker

global_64_128_256 = {'reuse_weights': False, 'batch_size': 16, 'device_id': 0, 
                     'which_gen': 'origin', 'which_disc':'global','dataset':'birds',
                     'g_lr': .0002/(2**0),'d_lr': .0002/(2**0), 'img_loss_ratio': 1, 'tune_img_loss':False,
                     'imsize':[64, 128,  256], 'load_from_epoch': 0, 'model_name':'gen_origin_disc_global', 
                     'reduce_dim_at':[8, 32, 128, 256]}

training_pool = np.array([
                 global_64_128_256
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

