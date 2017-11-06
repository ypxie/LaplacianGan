
D=256
name='zz_mmgan_plain_gl_disc_comG'

## if you pretrained from an outside model 
root=../Models
folder=${name}_coco_${D}   
mkdir -p ${root}/${folder}

CUDA_VISIBLE_DEVICES=${device} python zz_train_coco.py --dataset coco --batch_size 80 --imsize ${D} --model_name ${name} --ncritic_epoch_range 0 --epoch_decay 50 --gpus ${device} --which_gen comb_64_256 --which_disc comb_64_256 --save_freq 1 --KL_COE 2 --reuse_weights --load_from_epoch 6 | tee ${root}/${folder}/log.txt



#CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} --which_gen single_256 --which_disc single_256 --reuse_weights --load_from_epoch 500 --g_lr 0.00000625 --d_lr 0.00000625 | tee ${root}/${folder}/log.txt 

#CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} | tee ${root}/${folder}/log.txt
