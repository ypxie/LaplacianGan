import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..'))

home = os.path.expanduser('~')
proj_root = os.path.join('..')

data_root  = os.path.join(home, 'devbox', 'Shared_YZ', 'Results')
stack_root = os.path.join(home, 'devbox', 'Shared_YZ', 'StackGAN_visual_results')
#data_root = os.path.join(home, 'devbox', 'Shared_YZ', 'StackGAN_visual_results','Final')
#model_root = os.path.join(home, 'devbox', 'Shared_YZ', 'models')
model_root = os.path.join(proj_root, 'Models')

import torch.multiprocessing as mp
from LaplacianGan.proj_utils.local_utils import Indexflow
from LaplacianGan.neuralDist.test_nd_worker import test_worker

#/home/yuanpuxie/devbox/Shared_YZ/StackGAN_visual_results/Final/birds_test_large_samples_10copy_29330

if 1: #local [64, 256] 

    original_coco  =   \
                    { 'load_from_epoch': 24, 'batch_size': 8, 'device_id': 0,
                     "model_root": os.path.join(home,'ganData' ,'Models'),
                      "data_folder":os.path.join(data_root, 'coco','eval_nobugtesting_num_1'), 
                      "result_marker": "doesnotmatter", "dataset":"coco",
                      "file_name": "original.h5",
                      'model_name':'neural_dist_coco',
                    }

    zz_mmgan_plain_gl_disc_comG_coco_256_G_epoch_200  =   \
                    { 'load_from_epoch': 24, 'batch_size': 8, 'device_id': 0,
                      "model_root": os.path.join(home,'ganData' ,'Models'),
                      "data_folder":os.path.join(data_root, 'coco','eval_nobugtesting_num_1'), 
                      "result_marker": "doesnotmatter", "dataset":"coco",
                      "file_name": "zz_mmgan_plain_gl_disc_comG_coco_256_G_epoch_200.h5",
                      'model_name':'neural_dist_coco',
                    }

    # original_bird  =   \
    #                 { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 3,
    #                   "data_folder":os.path.join(data_root, 'birds','Finaltesting_num_10'), 
    #                   "result_marker": "testing_with_bugs_testing_num_10", "dataset":"birds",
    #                   "file_name": "original.h5",
    #                   'model_name':'neural_dist_birds',
    #                 }
    
    # original_flowers  =   \
    #                 { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 3,
    #                   "data_folder":os.path.join(data_root, 'flowers','Finaltesting_num_26'), 
    #                   "result_marker": "Finaltesting_num_26", "dataset":"flowers",
    #                   "file_name": "original.h5",
    #                   'model_name':'neural_dist_flowers',
    #                 }

    # neural_dist_birds_no_img_256  =   \
    #                 { 'load_from_epoch': 70, 'batch_size': 8, 'device_id': 2,
    #                   "data_folder":os.path.join(data_root, 'birds','testing_with_bugs_testing_num_10'), 
    #                   "result_marker": "testing_with_bugs_testing_num_10", "dataset":"birds",
    #                   "file_name": "gen_origin_disc_global_no_img_birds_[64, 128, 256]_G_epoch_501.h5",
    #                   'model_name':'neural_dist_birds',
    #                 }
    # neural_dist_birds_no_img_64_256  =   \
    #                 { 'load_from_epoch': 70, 'batch_size': 8, 'device_id': 2,
    #                   "data_folder":os.path.join(data_root, 'birds','testing_with_bugs_testing_num_10'), 
    #                   "result_marker": "testing_with_bugs_testing_num_10", "dataset":"birds",
    #                   "file_name": "gen_origin_disc_local_no_img_birds_[64, 256]_G_epoch_501.h5",
    #                   'model_name':'neural_dist_birds',
    #                 }
    
    neural_dist_birds_local_no_img_64_128_256  =   \
                    { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 0,
                      "data_folder":os.path.join(data_root, 'birds','testing_with_bugs_testing_num_10'), 
                      "result_marker": "testing_with_bugs_testing_num_10", "dataset":"birds",
                      "file_name": "gen_origin_disc_local_bug_no_img_birds_[64, 128, 256]_G_epoch_501.h5",
                      'model_name':'neural_dist_birds',
                    }

    # neural_dist_birds  =  \
    #                 { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 0,
    #                   "data_folder": os.path.join(data_root, 'birds','Finaltesting_num_10'), 
    #                   "result_marker": "Finaltesting_num_10", "dataset":"birds",
    #                   "file_name": "zz_mmgan_plain_local_512_ncit30_50decay_super4_res_2_birds_512_G_epoch_80.h5",
    #                   'model_name':'neural_dist_birds',
    #                 }

    # neural_dist_flowers  =   \
    #                 { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 2,
    #                   "data_folder":os.path.join(data_root, 'flowers','testing_with_bugs_testing_num_26'), 
    #                   "result_marker": "testing_with_bugs_testing_num_10", "dataset":"flowers",
    #                   "file_name": "gen_origin_disc_local_no_img_flowers_[64, 256]_G_epoch_501.h5",
    #                   'model_name':'neural_dist_flowers',
    #                 }
    # neural_dist_flowers_bug_64_128_256  =   \
    #                 { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 0,
    #                   "data_folder":os.path.join(data_root, 'flowers','testing_with_bugs_testing_num_26'), 
    #                   "result_marker": "testing_with_bugs_testing_num_10", "dataset":"flowers",
    #                   "file_name": "gen_origin_disc_local_bug_flowers_[64, 128, 256]_G_epoch_501.h5",
    #                   'model_name':'neural_dist_flowers',
    #                 }
    
    # final_256  =   \
    #                 { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 0,
    #                   "data_folder":os.path.join(data_root, 'flowers','Finaltesting_num_26'), 
    #                   "result_marker": "testing_with_bugs_testing_num_10", "dataset":"flowers",
    #                   "file_name": "zz_mmgan_plain_gl_disc_baldg2_flowers_256_G_epoch_580.h5",
    #                   'model_name':'neural_dist_flowers',
    #                 }
                   
    # neural_dist_flowers_bug_64_256  =   \
    #                 { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 2,
    #                   "data_folder":os.path.join(data_root, 'flowers','testing_with_bugs_testing_num_26'), 
    #                   "result_marker": "testing_with_bugs_testing_num_10", "dataset":"flowers",
    #                   "file_name": "gen_origin_disc_local_bug_flowers_[64, 256]_G_epoch_501",
    #                   'model_name':'neural_dist_flowers',
    #                 }
                              
    # stack_birds  =   \
    #                 { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 0, 
    #                   "data_folder":os.path.join(stack_root, 'Final', 'birds_test_large_samples_10copy_29330'),
    #                   "dataset":"birds",
    #                   "file_name": "64_256_results_29360.h5",
    #                   'model_name':'neural_dist_birds',
    #                 }

    # stack_flowers  =   \
    #                 { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 2, 
    #                   "data_folder":os.path.join(stack_root, 'Final', 'flowers_test_large_samples_26copy_30030'),
    #                   "dataset":"flowers",
    #                   "file_name": "64_256_results_30160.h5",
    #                   'model_name':'neural_dist_flowers',
    #                }

    stack_coco  =   \
                    { 'load_from_epoch': 37, 'batch_size': 16, 'device_id': 0,   #24
                      "data_folder":os.path.join(stack_root, 'coco_pytorch'),
                      "model_root" : os.path.join(home, 'ganData', 'Models'),
                      "dataset":"coco",
                      "file_name": "coco_stackgan_res.h5",
                      'model_name':'neural_dist_coco',
                    }
    

training_pool = np.array([
                    #neural_dist_birds_local_no_img_64_128_256,
                    stack_coco,
                    #original_coco,
                    #zz_mmgan_plain_gl_disc_comG_coco_256_G_epoch_200,
                    #final_256,
                    #neural_dist_flowers_bug_64_128_256, 
                    #neural_dist_birds,
                    #neural_dist_flowers_bug_64_128_256,
                    #neural_dist_flowers_bug_64_256   
                    #original_bird,
                    #original_flowers,
                    #neural_dist_birds_no_img_256,
                    #neural_dist_birds_no_img_64_256,
                    #neural_dist_birds,
                    #neural_dist_flowers
                    #stack_birds,
                    #stack_flowers
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

