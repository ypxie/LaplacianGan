import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..'))


data_root  = os.path.join( '..', 'Data')
model_root = os.path.join( '..', 'Models')

from train_worker import train_worker
import torch.multiprocessing as mp

from LaplacianGan.proj_utils.local_utils import Indexflow

large_global_local = {'reuse_weights': True, 'batch_size': 6, 'device_id': 1, 'gpu_list': [1,2,3], 
                   'imsize':256, 'load_from_epoch': 291, 'model_name':'large_global_local', 
                   'which_gen': 'origin', 'which_disc':'origin_global_local', 'dataset':'birds' }

training_pool = np.array([
                 large_global_local
                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

for select_ind in Indexflow(Totalnum, 2, random=False):
    select_pool = training_pool[select_ind]

    for this_dick in select_pool:

        p = mp.Process(target=train_worker, args= (data_root, model_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with', select_pool)

