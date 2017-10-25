
D=256
name='zz_mmgan_plain_gl_disc_ncric_comb128_256'

## if you pretrained from an outside model 
root=../Models
folder=${name}_birds_${D}   
mkdir -p ${root}/${folder}

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name}  --which_gen comb_128_256 --which_disc comb_128_256 --reuse_weights --load_from_epoch 280 --g_lr 0.00005 --d_lr 0.00005 | tee ${root}/${folder}/log.txt


#CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} --which_gen single_256 --which_disc single_256 --reuse_weights --load_from_epoch 500 --g_lr 0.00000625 --d_lr 0.00000625 | tee ${root}/${folder}/log.txt 

#CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} | tee ${root}/${folder}/log.txt
