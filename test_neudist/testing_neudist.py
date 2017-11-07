import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..'))

home = os.path.expanduser('~')
proj_root = os.path.join('..')

data_root = os.path.join(home, 'devbox', 'Shared_YZ', 'Results')

data_root = os.path.join(home, 'devbox', 'Shared_YZ', 'StackGAN_visual_results','Final')
#model_root = os.path.join(home, 'devbox', 'Shared_YZ', 'models')
model_root = os.path.join(proj_root, 'Models')

import torch.multiprocessing as mp
from LaplacianGan.proj_utils.local_utils import Indexflow
from LaplacianGan.neuralDist.test_nd_worker import test_worker

#/home/yuanpuxie/devbox/Shared_YZ/StackGAN_visual_results/Final/birds_test_large_samples_10copy_29330

if 1: #local [64, 256] 
    # neural_dist_birds  =   \
    #                 { 'load_from_epoch': 70, 'batch_size': 8, 'device_id': 2,
    #                   "data_folder":os.path.join(data_root, 'birds',testing_with_bugs_testing_num_10'), 
    #                   "result_marker": "testing_with_bugs_testing_num_10", "dataset":"birds",
    #                   "file_name": "zz_mmgan_plain_gl_disc_ncric_comb_64_256v2_birds_256_G_epoch_500.h5",
    #                   'model_name':'neural_dist_birds',
    #                 }

    stack_birds  =   \
                    { 'load_from_epoch': 70, 'batch_size': 8, 'device_id': 2, 
                      "data_folder":os.path.join(data_root, 'birds_test_large_samples_10copy_29330'),
                      "dataset":"birds",
                      "file_name": "64_256_results_29360.h5",
                      'model_name':'neural_dist_birds',
                    }

    stack_flowers  =   \
                    { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 2, 
                      "data_folder":os.path.join(data_root, 'flowers_test_large_samples_26copy_30030'),
                      "dataset":"flowers",
                      "file_name": "64_256_results_30160.h5",
                      'model_name':'neural_dist_flowers',
                    }

    

training_pool = np.array([
                    stack_birds,
                    stack_flowers
                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

for select_ind in Indexflow(Totalnum, 4, random=False):
    select_pool = training_pool[select_ind]

    for this_dick in select_pool:

        p = mp.Process(target=test_worker, args= (data_root, model_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with: ', select_ind)

