# flower
CUDA_VISIBLE_DEVICES=3 python inception_score.py --checkpoint_dir ./inception_finetuned_models/birds_valid299/model.ckpt --image_folder /data/data2/Shared_YZ/Results/birds/eval_nobugtesting_num_10 --h5_file 'gen_origin_disc_both_birds_[64, 128, 256]_G_epoch_540.h5' --batch_size 12
