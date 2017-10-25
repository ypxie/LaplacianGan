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
from LaplacianGan.test_worker import test_worker

save_spec = 'eval_nobug'


# gen_origin_disc_origin_flowers_all   = {'test_sample_num' : 26,  'load_from_epoch': 396, 'dataset':'flowers', 
#                 'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_origin_flowers_[64, 128, 256]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
#                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

# gen_origin_disc_both_flowers_all   = {'test_sample_num' : 26,  'load_from_epoch': 387, 'dataset':'flowers', 
#                 'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_both_flowers_[64, 128, 256]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
#                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }


# gen_origin_disc_both_birds_all   = {'test_sample_num' : 10,  'load_from_epoch': 405, 'dataset':'birds', 
#                 'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_both_birds_[64, 128, 256]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
#                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

# gen_origin_disc_global_no_img_birds  =   \
#                {'test_sample_num' : 10,  'load_from_epoch': 597, 'dataset':'birds', 
#                 'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_global_no_img_birds_[64, 128, 256]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
#                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

# gen_origin_disc_both_birds  =   \
#                {'test_sample_num' : 10,  'load_from_epoch': 432, 'dataset':'birds', 
#                 'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_both_birds_[64, 128, 256]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
#                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }


gen_origin_disc_both_birds_561   = {'batch_size': 8, 'device_id': 0,'imsize':256, 'load_from_epoch': 561, 'train_mode': False,
                'model_name':'gen_origin_disc_both_birds_[64, 128, 256]', 'save_spec': save_spec, 'test_sample_num' : 10,
                'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

gen_origin_disc_both_birds_540  = {'batch_size': 8, 'device_id': 1,'imsize':256, 'load_from_epoch': 540, 'train_mode': False,
                'model_name':'gen_origin_disc_both_birds_[64, 128, 256]', 'save_spec': save_spec, 'test_sample_num' : 10,
                'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

gen_origin_disc_both_birds_510  = {'batch_size': 8, 'device_id': 2,'imsize':256, 'load_from_epoch': 510, 'train_mode': False,
                'model_name':'gen_origin_disc_both_birds_[64, 128, 256]', 'save_spec': save_spec, 'test_sample_num' : 10,
                'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }


training_pool = np.array([
                gen_origin_disc_both_birds_561,
                gen_origin_disc_both_birds_540,
                gen_origin_disc_both_birds_510
                #gen_origin_disc_global_no_img_birds,
                #gen_origin_disc_both_birds
                 #gen_origin_disc_origin_flowers_all,
                 #gen_origin_disc_both_flowers_all,
                 #gen_origin_disc_both_birds_all
                 #flower_plain_gl_disc_ncric_540,
                 #flower_plain_gl_disc_ncric_560,
                 #flower_plain_gl_disc_ncric_580
                 #flower_plain_gl_disc_allncric,
                 #flower_plain_gl_disc_ncric,
                 #bird_plain_gl_disc_ncric_comb_64_256v2,
                 #bird_plain_gl_disc_ncric_single_256,
                 #bird_plain_gl_disc_birds
                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

for select_ind in Indexflow(Totalnum, 3, random=False):
    select_pool = training_pool[select_ind]

    for this_dick in select_pool:

        p = mp.Process(target=test_worker, args= (data_root, model_root, save_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with: ', select_ind)

