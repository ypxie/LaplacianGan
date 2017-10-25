import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..'))

home = os.path.expanduser('~')
proj_root = os.path.join('..')
data_root = os.path.join(proj_root, 'Data')

model_root = os.path.join(home, 'devbox', 'Shared_YZ', 'models')
save_root  =  os.path.join(home, 'devbox', 'Shared_YZ', 'Results')

import torch.multiprocessing as mp
from LaplacianGan.proj_utils.local_utils import Indexflow
from LaplacianGan.test_worker_coco import test_worker

save_spec = 'eval_nobug'

gen_origin_disc_origin_coco_64  =   \
               {'test_sample_num' : 1,  'load_from_epoch': 261, 'dataset':'coco', 
                'device_id': 0,'imsize':[64], 'model_name':'gen_origin_disc_origin_coco_[64]',
                'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
                 'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }

# gen_origin_disc_origin_coco_64_128  =   \
#                {'test_sample_num' : 1,  'load_from_epoch': 114, 'dataset':'coco', 
#                 'device_id': 0,'imsize':[64, 128], 'model_name':'gen_origin_disc_origin_coco_[64, 128]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
#                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }


training_pool = np.array([
                 gen_origin_disc_origin_coco_64,
                 #gen_origin_disc_origin_coco_64_128
                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

#test_worker(data_root, model_root, save_root, gen_origin_disc_origin_coco_64)

for select_ind in Indexflow(Totalnum, 3, random=False):
    select_pool = training_pool[select_ind]

    for this_dick in select_pool:

        p = mp.Process(target=test_worker, args= (data_root, model_root, save_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with: ', select_ind)

