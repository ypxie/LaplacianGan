
D=512
name='zz_mmgan_plain_gl_512'

## if you pretrained from an outside model 
root=../Models
folder=${name}_birds_${D}   
mkdir -p ${root}/${folder}

# cp -v ../Models/zz_mmgan_plain_gl_disc_birds_256/G_epoch500.pth ${root}/${folder}/G_epoch_256_init.pth

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 10 --imsize ${D} --model_name ${name} --which_gen super --which_disc super --reuse_weights --load_from_epoch 500 --g_lr 0.000025 --d_lr 0.000025 --load_from_epoch 300 | tee ${root}/${folder}/log.txt
