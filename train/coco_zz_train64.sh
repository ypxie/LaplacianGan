
D=64
name='zz_mmgan_plain_gl_disc_64'

## if you pretrained from an outside model 
root=../Models
folder=${name}_coco_${D}   
mkdir -p ${root}/${folder}

python zz_train_coco.py --dataset coco --batch_size 128 --imsize ${D} --model_name ${name} --ncritic_epoch_range 0 --epoch_decay 60 --gpus ${device} --save_freq 1 | tee ${root}/${folder}/log.txt



#CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} --which_gen single_256 --which_disc single_256 --reuse_weights --load_from_epoch 500 --g_lr 0.00000625 --d_lr 0.00000625 | tee ${root}/${folder}/log.txt 

#CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} | tee ${root}/${folder}/log.txt
