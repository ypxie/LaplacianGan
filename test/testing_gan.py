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

save_spec = 'eval_bs_1'

# flower_400   = {'batch_size': 8, 'device_id': 0,'imsize':256, 'load_from_epoch': 400, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_baldg2_flowers_256', 'save_spec': save_spec, 'test_sample_num' : 26,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'flowers','reduce_dim_at':[8, 32, 128, 256] }

# flower_500 = flower_400.copy()
# flower_500['load_from_epoch'] = 500
# flower_500['device_id'] = 1

# flower_580 = flower_400.copy()
# flower_580['load_from_epoch'] = 580
# flower_580['device_id'] = 0

# birds_300 = {'batch_size': 8, 'device_id': 1,'imsize':256, 'load_from_epoch': 300, 'train_mode': False,
#              'model_name':'zz_mmgan_plain_gl_disc_continue_ncric_birds_256', 'save_spec': save_spec, 
#              'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }
# birds_400 = birds_300.copy()
# birds_400['load_from_epoch'] = 400
# birds_400['device_id'] = 0

# birds_500 = birds_300.copy()
# birds_500['load_from_epoch'] = 500
# birds_400['device_id'] = 1

# training_pool = np.array([
#                  flower_400, flower_500, flower_580,
#                  #birds_300, birds_400, birds_500
#                  ])

# flower_plain_gl_disc_allncric   = {'batch_size': 8, 'device_id': 1,'imsize':256, 'load_from_epoch': 500, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_allncric_flowers_256', 'save_spec': save_spec, 'test_sample_num' : 26,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'flowers','reduce_dim_at':[8, 32, 128, 256] }

# flower_plain_gl_disc_ncric   = {'batch_size': 8, 'device_id': 0,'imsize':256, 'load_from_epoch': 500, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_ncric_flowers_256', 'save_spec': save_spec, 'test_sample_num' : 26,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'flowers','reduce_dim_at':[8, 32, 128, 256] }

# bird_plain_gl_disc_ncric_comb_64_256v2   = {'batch_size': 8, 'device_id': 1,'imsize':[64, 256], 'load_from_epoch': 500, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_ncric_comb_64_256v2_birds_256', 'save_spec': save_spec, 'test_sample_num' : 10,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

# bird_plain_gl_disc_ncric_single_256   = {'batch_size': 8, 'device_id': 0, 'imsize':[256], 'load_from_epoch': 500, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_ncric_single_256_birds_256', 'save_spec': save_spec, 'test_sample_num' : 10,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

# bird_plain_gl_disc_birds  = {'batch_size': 8, 'device_id': 0,'imsize':256, 'load_from_epoch': 500, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_birds_256', 'save_spec': save_spec, 'test_sample_num' : 10,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }


# flower_plain_gl_disc_ncric_540   = {'batch_size': 8, 'device_id': 0,'imsize':256, 'load_from_epoch': 540, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_ncric_flowers_256', 'save_spec': save_spec, 'test_sample_num' : 26,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'flowers','reduce_dim_at':[8, 32, 128, 256] }

# flower_plain_gl_disc_ncric_560   = {'batch_size': 8, 'device_id': 0,'imsize':256, 'load_from_epoch': 560, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_ncric_flowers_256', 'save_spec': save_spec, 'test_sample_num' : 26,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'flowers','reduce_dim_at':[8, 32, 128, 256] }

# flower_plain_gl_disc_ncric_580   = {'batch_size': 8, 'device_id': 0,'imsize':256, 'load_from_epoch': 580, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_ncric_flowers_256', 'save_spec': save_spec, 'test_sample_num' : 26,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'flowers','reduce_dim_at':[8, 32, 128, 256] }

# birds_plain_gl_disc_ncric_580   = {'batch_size': 8, 'device_id': 0,'imsize':256, 'load_from_epoch': 500, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_disc_ncric_fulglo_256_birds_256', 'save_spec': save_spec, 'test_sample_num' : 10,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

gen_origin_disc_both_birds_561   = {'batch_size': 8, 'device_id': 3,'imsize':256, 'load_from_epoch': 561, 'train_mode': False,
                'model_name':'gen_origin_disc_both_birds_[64, 128, 256]', 'save_spec': save_spec, 'test_sample_num' : 10,
                'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

gen_origin_disc_both_birds_540  = {'batch_size': 8, 'device_id': 3,'imsize':256, 'load_from_epoch': 540, 'train_mode': False,
                'model_name':'gen_origin_disc_both_birds_[64, 128, 256]', 'save_spec': save_spec, 'test_sample_num' : 10,
                'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

gen_origin_disc_both_birds_510  = {'batch_size': 8, 'device_id': 3,'imsize':256, 'load_from_epoch': 510, 'train_mode': False,
                'model_name':'gen_origin_disc_both_birds_[64, 128, 256]', 'save_spec': save_spec, 'test_sample_num' : 10,
                'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }



training_pool = np.array([
                gen_origin_disc_both_birds_561,
                gen_origin_disc_both_birds_540,
                gen_origin_disc_both_birds_510
                 #birds_plain_gl_disc_ncric_580,
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

