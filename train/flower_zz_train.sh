
D=256
name='zz_mmgan_plain_gl_disc_ncric'

## if you pretrained from an outside model 
root=../Models
folder=${name}_flowers_${D}
mkdir -p ${root}/${folder}

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset flowers --batch_size 16 --imsize ${D} --model_name ${name} --reuse_weights --load_from_epoch 100 --g_lr 0.0001 --d_lr 0.0001 | tee ${root}/${folder}/log.txt

