import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..'))

home = os.path.expanduser("~")
#data_root  = os.path.join('..', 'Data')
#model_root = os.path.join( '..', 'Models')
data_root  = os.path.join(home, 'ganData')
model_root = os.path.join(data_root, 'Models')

import torch.multiprocessing as mp
from LaplacianGan.proj_utils.local_utils import Indexflow
from LaplacianGan.neuralDist.train_nd_worker_coco import train_worker

# local_global disc. We test both large and small model
# 101 201 301 401 501 601
# 2   4   8   16  32  64

coco_neudist  = { 'reuse_weights': True, 'batch_size': 64, 'device_id': 0, 'lr': .0002/(2**0), 'margin':0.05,
                  'imsize':[64, 256], 'load_from_epoch': 24, 'model_name':'neural_dist', 
                  'dataset':'coco', "save_freq":1,
                }

training_pool = np.array([
                 coco_neudist
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
