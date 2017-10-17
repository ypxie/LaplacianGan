
D=256
name='zz_mmgan_plain_gl_disc_ncric_comb_64_256'

## if you pretrained from an outside model 
root=../Models
folder=${name}_birds_${D}   
mkdir -p ${root}/${folder}

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} --which_gen comb_64_256 --which_dis comb_64_256 --reuse_weights --load_from_epoch 100 --g_lr 0.0001 --d_lr 0.0001 | tee ${root}/${folder}/log.txt

# CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} --which_gen single_256 --which_dis single_256 --reuse_weights --load_from_epoch 100 --g_lr 0.0001 --d_lr 0.0001 | tee ${root}/${folder}/log.txt 

#CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} | tee ${root}/${folder}/log.txt
