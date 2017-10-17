# flower
CUDA_VISIBLE_DEVICES=0 python inception_score.py --checkpoint_dir ./inception_finetuned_models/flowers_valid299/model.ckpt --num_classes 20 --image_folder /data/data2/Shared_YZ/Results/flowers/eval_bs_1testing_num_26 --h5_file zz_mmgan_plain_gl_disc_baldg2_flowers_256_G_epoch_580.h5
