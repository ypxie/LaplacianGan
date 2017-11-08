# flower
CUDA_VISIBLE_DEVICES=0 python inception_score.py --checkpoint_dir ./inception_finetuned_models/flowers_valid299/model.ckpt --num_classes 20 --batch_size 8 --image_folder '/data/data2/Shared_YZ/Results/flowers/eval_nobugtesting_num_26' --h5_file 'gen_origin_disc_local_flowers_[64, 256]_G_epoch_597.h5'
