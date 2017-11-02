# flower
CUDA_VISIBLE_DEVICES=6 python inception_score.py --checkpoint_dir ./inception_finetuned_models/birds_valid299/model.ckpt --image_folder /data/data2/Shared_YZ/Results/birds/eval_nobugtesting_num_10 --h5_file 'gen_origin_disc_local_low_birds_[64, 128, 256]_G_epoch_501.h5' --batch_size 32
