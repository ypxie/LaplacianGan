
D=256
name='zz_mmgan_plain_gl_disc_ncric10'

## if you pretrained from an outside model 
root=../Models
folder=${name}_flowers_${D}
mkdir -p ${root}/${folder}

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset flowers --batch_size 16 --imsize ${D} --model_name ${name} --ncritic_epoch_range 10 --reuse_weights --load_from_epoch 340 --g_lr 0.000025 --d_lr 0.000025| tee ${root}/${folder}/log.txt

