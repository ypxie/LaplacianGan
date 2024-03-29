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

large_global_local = {'reuse_weights': False, 'batch_size': 12, 'device_id': 2, 'gpu_list': [0], 
                    'imsize':[256], 'load_from_epoch': 0, 'model_name':'gen_origin_disc_origin', 
                    'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256]}

training_pool = np.array([
                 #large_global_local_bigger,
                 large_global_local
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

