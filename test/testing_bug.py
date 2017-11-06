import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..'))

home = os.path.expanduser('~')
proj_root = os.path.join('..')
data_root = os.path.join(proj_root, 'Data')
#data_root = os.path.join(home, 'ganData')

model_root = os.path.join(home, 'devbox', 'Shared_YZ', 'models')
#model_root = os.path.join(proj_root, 'Models')
#model_root = os.path.join(data_root, 'Models')

save_root  =  os.path.join(home, 'devbox', 'Shared_YZ', 'Results')

import torch.multiprocessing as mp
from LaplacianGan.proj_utils.local_utils import Indexflow
from LaplacianGan.test_worker import test_worker

save_spec = 'testing_with_bugs_'

if 1: #local [64, 256]

    zz_mmgan_plain_gl_disc_ncric_comb_64_256v2_birds_256_500  =   \
                   { 'test_sample_num' : 10,  'load_from_epoch': 500, 'dataset':'birds', "save_images":True,
                     'device_id': 0,'imsize':[64, 256], 'model_name':'zz_mmgan_plain_gl_disc_ncric_comb_64_256v2_birds_256',
                     'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
                     'which_disc':'local', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_local_flowers_501  =   \
    #                { 'test_sample_num' : 26,  'load_from_epoch': 501, 'dataset':'flowers', "save_images":True,
    #                  'device_id': 1,'imsize':[64, 256], 'model_name':'gen_origin_disc_local_flowers_[64, 256]',
    #                  'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'local', 'reduce_dim_at':[8, 32, 128, 256] }

training_pool = np.array([
                zz_mmgan_plain_gl_disc_ncric_comb_64_256v2_birds_256_500,

                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

for select_ind in Indexflow(Totalnum, 4, random=False):
    select_pool = training_pool[select_ind]

    for this_dick in select_pool:

        p = mp.Process(target=test_worker, args= (data_root, model_root, save_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with: ', select_ind)

