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

# local_global disc. We test both large and small model
# 101 201 301 401 501 601
# 2   4   8   16  32  64

local_no_img_64_128_256      = { 'reuse_weights': True, 'batch_size': 16, 'device_id': 0,  'use_img_loss' : False,
                        'which_gen': 'origin', 'which_disc':'local', 'g_lr': .0002/(2**3),'d_lr': .0002/(2**3), 
                        'imsize':[64,128, 256], 'load_from_epoch': 336, 'model_name':'gen_origin_disc_local_no_img', 
                        'dataset':'flowers',
                        'reduce_dim_at':[8, 32, 128, 256]}

training_pool = np.array([
                 local_no_img_64_128_256
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

