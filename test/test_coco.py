import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..'))

home = os.path.expanduser('~')
proj_root = os.path.join('..')

data_root = os.path.join(proj_root, 'Data')
#data_root = os.path.join(home, 'ganData')


model_root = os.path.join(home, 'devbox', 'Shared_YZ', 'models')
save_root  =  os.path.join(home, 'devbox', 'Shared_YZ', 'Results')

import torch.multiprocessing as mp
from LaplacianGan.proj_utils.local_utils import Indexflow
from LaplacianGan.test_worker_coco import test_worker

save_spec = 'eval_nobug'

# 261
# gen_origin_disc_origin_coco_64  =   \
#                {'test_sample_num' : 1,  'load_from_epoch': 120, 'dataset':'coco', 
#                 'device_id': 0,'imsize':[64], 'model_name':'gen_origin_disc_origin_coco_[64]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 64, 'which_gen': 'origin',
#                 'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }

# gen_origin_disc_origin_coco_128  =   \
#                {'test_sample_num' : 1,  'load_from_epoch': 99, 'dataset':'coco', 
#                 'device_id': 0,'imsize':[64, 128], 'model_name':'gen_origin_disc_origin_coco_[64, 128]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 16, 'which_gen': 'origin',
#                 'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }

# gen_origin_disc_origin_coco_128_114  =   \
#                {'test_sample_num' : 1,  'load_from_epoch': 114, 'dataset':'coco', 
#                 'device_id': 0,'imsize':[64, 128], 'model_name':'gen_origin_disc_origin_coco_[64, 128]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 16, 'which_gen': 'origin',
#                 'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }


# gen_origin_disc_origin_coco_64_128  =   \
#                {'test_sample_num' : 1,  'load_from_epoch': 114, 'dataset':'coco', 
#                 'device_id': 0,'imsize':[64, 128], 'model_name':'gen_origin_disc_origin_coco_[64, 128]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
#                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }

parap_64_256_101  =   \
               {'test_sample_num' : 1,  'load_from_epoch': 101, 'dataset':'coco', "num_resblock":1,
                'device_id': 0,'imsize':[64, 256], 'model_name':'zz_mmgan_plain_gl_disc_comG_coco_256',
                'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
                 'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }

parap_64_256_100  =   \
               {'test_sample_num' : 1,  'load_from_epoch': 100, 'dataset':'coco', "num_resblock":1,
                'device_id': 0,'imsize':[64, 256], 'model_name':'zz_mmgan_plain_gl_disc_comG_coco_256',
                'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
                 'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }

parap_64_256_50  =   \
               {'test_sample_num' : 1,  'load_from_epoch': 50, 'dataset':'coco', "num_resblock":1,
                'device_id': 0,'imsize':[64, 256], 'model_name':'zz_mmgan_plain_gl_disc_comG_coco_256',
                'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
                 'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }

parap_64_256_80  =   \
               {'test_sample_num' : 1,  'load_from_epoch': 80, 'dataset':'coco', "num_resblock":1,
                'device_id': 0,'imsize':[64, 256], 'model_name':'zz_mmgan_plain_gl_disc_comG_coco_256',
                'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
                 'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256], 'save_images':True }

training_pool = np.array([
                    #parap_64_256_101,
                    parap_64_256_100,
                    parap_64_256_50,
                    parap_64_256_80,
                 #gen_origin_disc_origin_coco_64,
                 #gen_origin_disc_origin_coco_128,
                 #gen_origin_disc_origin_coco_128_114
                 #gen_origin_disc_origin_coco_64_128
                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

#test_worker(data_root, model_root, save_root, gen_origin_disc_origin_coco_64)

for select_ind in Indexflow(Totalnum, 2, random=False):
    select_pool = training_pool[select_ind]

    for this_dick in select_pool:

        p = mp.Process(target=test_worker, args= (data_root, model_root, save_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with: ', select_ind)

