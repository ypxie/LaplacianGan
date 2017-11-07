import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..'))

home = os.path.expanduser('~')
proj_root = os.path.join('..')
#data_root = os.path.join(proj_root, 'Data')
data_root = os.path.join(home, 'ganData')

#model_root = os.path.join(home, 'devbox', 'Shared_YZ', 'models')
#model_root = os.path.join(proj_root, 'Models')
model_root = os.path.join(data_root, 'Models')

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


# gen_origin_disc_both_birds  =   \
#                {'test_sample_num' : 10,  'load_from_epoch': 432, 'dataset':'birds', 
#                 'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_both_birds_[64, 128, 256]',
#                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
#                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }


# gen_origin_disc_both_birds_561   = {'batch_size': 8, 'device_id': 0,'imsize':256, 'load_from_epoch': 561, 'train_mode': False,
#                 'model_name':'gen_origin_disc_both_birds_[64, 128, 256]', 'save_spec': save_spec, 'test_sample_num' : 10,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

# gen_origin_disc_both_birds_540  = {'batch_size': 8, 'device_id': 1,'imsize':256, 'load_from_epoch': 540, 'train_mode': False,
#                 'model_name':'gen_origin_disc_both_birds_[64, 128, 256]', 'save_spec': save_spec, 'test_sample_num' : 10,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

# gen_origin_disc_both_birds_510  = {'batch_size': 8, 'device_id': 2,'imsize':256, 'load_from_epoch': 510, 'train_mode': False,
#                 'model_name':'gen_origin_disc_both_birds_[64, 128, 256]', 'save_spec': save_spec, 'test_sample_num' : 10,
#                 'which_gen': 'origin', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }


# zz_mmgan_plain_gl_512_birds_512_460  = {'batch_size': 8, 'device_id': 6,'imsize':256, 'load_from_epoch': 460, 'train_mode': False,
#                 'model_name':'zz_mmgan_plain_gl_512_birds_512', 'save_spec': save_spec, 'test_sample_num' : 10,
#                 'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

# zz_birds_512_res_3_80  = {'batch_size': 4, 'device_id': 0,'imsize':256, 'load_from_epoch': 80, 'train_mode': False,
#                           'model_name':'zz_mmgan_plain_local_512_ncit30_50decay_super3_res_3_birds_512',
#                           'save_spec': save_spec, 'test_sample_num' : 10,       "num_resblock":3,
#                           'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

# zz_birds_512_res_2_80  = {'batch_size': 4, 'device_id': 1,'imsize':256, 'load_from_epoch': 80, 'train_mode': False,
#                           'model_name':'zz_mmgan_plain_local_512_ncit30_50decay_super4_res_2_birds_512',
#                           'save_spec': save_spec, 'test_sample_num' : 10,       "num_resblock":2,
#                           'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

# zz_birds_512_res_2_150  = {'batch_size': 2, 'device_id': 0,'imsize':256, 'load_from_epoch': 150, 'train_mode': False,
#                           'model_name':'zz_mmgan_plain_local_512_ncit30_50decay_super4_res_2_birds_512',
#                           'save_spec': save_spec, 'test_sample_num' : 10,       "num_resblock":2,
#                           'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

# zz_birds_512_res_2_180  = {'batch_size': 2, 'device_id': 0,'imsize':256, 'load_from_epoch': 180, 'train_mode': False,
#                           'model_name':'zz_mmgan_plain_local_512_ncit30_50decay_super4_res_2_birds_512',
#                           'save_spec': save_spec, 'test_sample_num' : 10,       "num_resblock":2,
#                           'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }


# zz_mmgan_plain_local_512_ncit0_20decay_res3_img2_nopair_birds_512_50 = \
#                          {'batch_size': 2, 'device_id': 0,'imsize':256, 'load_from_epoch': 50, 'train_mode': False,
#                           'model_name':'zz_mmgan_plain_local_512_ncit0_20decay_res2_img2_nopair_birds_512',
#                           'save_spec': save_spec, 'test_sample_num' : 10,       "num_resblock":2,
#                           'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

# zz_mmgan_plain_local_512_ncit0_20decay_res3_img2_nopair_birds_512_99 = \
#                          {'batch_size': 2, 'device_id': 1,'imsize':256, 'load_from_epoch': 99, 'train_mode': False,
#                           'model_name':'zz_mmgan_plain_local_512_ncit0_20decay_res2_img2_nopair_birds_512',
#                           'save_spec': save_spec, 'test_sample_num' : 10,       "num_resblock":2,
#                           'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }



if 1:
    data_root = os.path.join(home, 'ganData')
    model_root = os.path.join(home, 'devbox', 'Shared_YZ', 'models')
    
    zz_mmgan_plain_local_512_ncit0_10decay_res3_pw2_birds_512_20 = \
                         {'batch_size': 2, 'device_id': 0,'imsize':256, 'load_from_epoch': 20, 'train_mode': False,
                          'model_name':'zz_mmgan_plain_local_512_ncit0_10decay_res3-pw2_birds_512',
                          'save_spec': save_spec, 'test_sample_num' : 10,       "num_resblock":3,
                          'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }
    
    zz_mmgan_plain_local_512_ncit0_10decay_res3_pw2_birds_512_40 = \
                         {'batch_size': 2, 'device_id': 0,'imsize':256, 'load_from_epoch': 40, 'train_mode': False,
                          'model_name':'zz_mmgan_plain_local_512_ncit0_10decay_res3-pw2_birds_512',
                          'save_spec': save_spec, 'test_sample_num' : 10,       "num_resblock":3,
                          'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }

    zz_mmgan_plain_local_512_ncit0_10decay_res3_pw2_birds_512_60 = \
                         {'batch_size': 2, 'device_id': 0,'imsize':256, 'load_from_epoch': 60, 'train_mode': False,
                          'model_name':'zz_mmgan_plain_local_512_ncit0_10decay_res3-pw2_birds_512',
                          'save_spec': save_spec, 'test_sample_num' : 10,       "num_resblock":3,
                          'which_gen': 'super', 'which_disc':'origin', 'dataset':'birds','reduce_dim_at':[8, 32, 128, 256] }


if 0: #local flower and bird
    pass
    #---Following 4 has been done------
    # gen_origin_disc_global_local_birds_597  =   \
    #                { 'test_sample_num' : 10,  'load_from_epoch': 597, 'dataset':'birds', 
    #                  'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_local_birds_[64, 128, 256]',
    #                  'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_global_local_birds_501  =   \
    #                { 'test_sample_num' : 10,  'load_from_epoch': 501, 'dataset':'birds', 
    #                  'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_local_birds_[64, 128, 256]',
    #                  'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_global_local_flower_597  =   \
    #                { 'test_sample_num' : 26,  'load_from_epoch': 597, 'dataset':'flowers', 
    #                  'device_id': 1,'imsize':256, 'model_name':'gen_origin_disc_local_flowers_[64, 128, 256]',
    #                  'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_global_local_flower_501  =   \
    #                { 'test_sample_num' : 26,  'load_from_epoch': 501, 'dataset':'flowers', 
    #                  'device_id': 1,'imsize':256, 'model_name':'gen_origin_disc_local_flowers_[64, 128, 256]',
    #                  'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }
#-----------------------------------------------------------
if 0: # local low image and flower
    "-------Following not been done yet.------------------------"
    pass
    # gen_origin_disc_global_local_low_birds_597  =   \
    #                { 'test_sample_num' : 10,  'load_from_epoch': 597, 'dataset':'birds', 
    #                  'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_local_low_birds_[64, 128, 256]',
    #                  'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_global_local_low_birds_501  =   \
    #                { 'test_sample_num' : 10,  'load_from_epoch': 501, 'dataset':'birds', 
    #                  'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_local_low_birds_[64, 128, 256]',
    #                  'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_global_local_low_flower_597  =   \
    #                { 'test_sample_num' : 26,  'load_from_epoch': 597, 'dataset':'flowers', 
    #                  'device_id': 1,'imsize':256, 'model_name':'gen_origin_disc_local_low_flowers_[64, 128, 256]',
    #                  'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_global_local_low_flower_501  =   \
    #                { 'test_sample_num' : 26,  'load_from_epoch': 501, 'dataset':'flowers', 
    #                  'device_id': 1,'imsize':256, 'model_name':'gen_origin_disc_local_low_flowers_[64, 128, 256]',
    #                  'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }
    
if 0:  # no image bird and flower
    "------------------The following no image bird---------- already done---------"
    pass
    # gen_origin_disc_global_no_img_birds_498  =   \
    #                {'test_sample_num' : 10,  'load_from_epoch': 498, 'dataset':'birds', 
    #                 'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_global_no_img_birds_[64, 128, 256]',
    #                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_global_no_img_birds_591  =   \
    #                {'test_sample_num' : 10,  'load_from_epoch': 591, 'dataset':'birds', 
    #                 'device_id': 1,'imsize':256, 'model_name':'gen_origin_disc_global_no_img_birds_[64, 128, 256]',
    #                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_local_no_img_flowers_501  =   \
    #                {'test_sample_num' : 26,  'load_from_epoch': 501, 'dataset':'flowers', 
    #                 'device_id': 0,'imsize':256, 'model_name':'gen_origin_disc_local_no_img_flowers_[64, 128, 256]',
    #                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

    # gen_origin_disc_local_no_img_flowers_597  =   \
    #                {'test_sample_num' : 26,  'load_from_epoch': 597, 'dataset':'flowers', 
    #                 'device_id': 1,'imsize':256, 'model_name':'gen_origin_disc_local_no_img_flowers_[64, 128, 256]',
    #                 'train_mode': False,  'save_spec': save_spec, 'batch_size': 8, 'which_gen': 'origin',
    #                  'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }


if 0: # no img loss [64, 256]
    data_root = os.path.join(home, 'ganData')
    model_root = os.path.join(data_root, 'Models')

    gen_origin_disc_local_no_img_birds_597  =   \
                   { 'test_sample_num' : 10,  'load_from_epoch': 597, 'dataset':'birds', "save_images":True, 
                     'device_id': 0,'imsize':[64, 256], 'model_name':'gen_origin_disc_local_no_img_birds_[64, 256]',
                     'train_mode': False,  'save_spec': save_spec, 'batch_size': 2, 'which_gen': 'origin',
                     'which_disc':'local', 'reduce_dim_at':[8, 32, 128, 256] }

    gen_origin_disc_local_no_img_birds_501  =   \
                   { 'test_sample_num' : 10,  'load_from_epoch': 501, 'dataset':'birds', "save_images":True,
                     'device_id': 1,'imsize':[64, 256], 'model_name':'gen_origin_disc_local_no_img_birds_[64, 256]',
                     'train_mode': False,  'save_spec': save_spec, 'batch_size': 2, 'which_gen': 'origin',
                     'which_disc':'local', 'reduce_dim_at':[8, 32, 128, 256] }

    gen_origin_disc_local_no_img_flowers_501  =   \
                   { 'test_sample_num' : 26,  'load_from_epoch': 501, 'dataset':'flowers', "save_images":True,
                     'device_id': 1, 'imsize':[64, 256], 'model_name':'gen_origin_disc_local_no_img_flowers_[64, 256]',
                     'train_mode': False,  'save_spec': save_spec, 'batch_size': 2, 'which_gen': 'origin',
                     'which_disc':'local', 'reduce_dim_at':[8, 32, 128, 256] }

    gen_origin_disc_local_no_img_flowers_597  =   \
                   { 'test_sample_num' : 26,  'load_from_epoch': 597, 'dataset':'flowers', "save_images":True,
                     'device_id': 0,'imsize':[64, 256], 'model_name':'gen_origin_disc_local_no_img_flowers_[64, 256]',
                     'train_mode': False,  'save_spec': save_spec, 'batch_size': 2, 'which_gen': 'origin',
                     'which_disc':'local', 'reduce_dim_at':[8, 32, 128, 256] }
    
    
training_pool = np.array([
                zz_mmgan_plain_local_512_ncit0_10decay_res3_pw2_birds_512_20,
                zz_mmgan_plain_local_512_ncit0_10decay_res3_pw2_birds_512_40,
                zz_mmgan_plain_local_512_ncit0_10decay_res3_pw2_birds_512_60
                
                #gen_origin_disc_local_no_img_birds_597,
                #gen_origin_disc_local_no_img_birds_501,
                #gen_origin_disc_local_no_img_flowers_597,
                #gen_origin_disc_local_no_img_flowers_501

                #zz_mmgan_plain_local_512_ncit0_20decay_res3_img2_nopair_birds_512_50,
                #zz_mmgan_plain_local_512_ncit0_20decay_res3_img2_nopair_birds_512_99
                #zz_birds_512_res_2_150,
                #zz_birds_512_res_2_180
                #gen_origin_disc_local_no_img_flowers_501,
                #gen_origin_disc_local_no_img_flowers_597
                #gen_origin_disc_global_local_low_birds_597,
                #gen_origin_disc_global_local_low_birds_501,
                #gen_origin_disc_global_local_low_flower_597,
                #gen_origin_disc_global_local_low_flower_501
                #gen_origin_disc_global_no_img_birds_498,
                #gen_origin_disc_global_no_img_birds_591
                # gen_origin_disc_global_local_birds_597,
                # gen_origin_disc_global_local_birds_501,
                # gen_origin_disc_global_local_flower_597,
                # gen_origin_disc_global_local_flower_501
                #zz_mmgan_plain_gl_512_birds_512_460
                #gen_origin_disc_both_birds_561,
                #gen_origin_disc_both_birds_540,
                #gen_origin_disc_both_birds_510
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

for select_ind in Indexflow(Totalnum, 4, random=False):
    select_pool = training_pool[select_ind]

    for this_dick in select_pool:

        p = mp.Process(target=test_worker, args= (data_root, model_root, save_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with: ', select_ind)

