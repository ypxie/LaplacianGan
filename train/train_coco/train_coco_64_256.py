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
coco_64_256 = {'reuse_weights': True, 'batch_size': 12, 'device_id': 0, 'gpu_list': [0], 
               'KL_COE':0.01,
               'img_loss_ratio':0.5/(2**2), 'tune_img_loss':True, 'g_lr': .0002/(2**2),  'd_lr': .0002/(2**2), 
                'imsize':[64, 256], 'load_from_epoch': 30, 'model_name':'gen_origin_disc_local', 
                'which_gen': 'origin', 'which_disc':'origin', 'dataset':'coco',
                'reduce_dim_at':[8, 32, 128, 256], 'num_resblock':2 }

training_pool = np.array([
                 #coco_64,
                  coco_64_256,
                  #coco_256_fine
                 #coco_128,
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

