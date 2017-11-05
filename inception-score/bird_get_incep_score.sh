# flower
CUDA_VISIBLE_DEVICES=3 python inception_score.py --checkpoint_dir ./inception_finetuned_models/birds_valid299/model.ckpt --image_folder /data/data2/Shared_YZ/Results/birds/eval_nobugtesting_num_10 --h5_file 'zz_mmgan_plain_local_512_ncit30_50decay_super4_res_2_birds_512_G_epoch_80.h5' --batch_size 32
