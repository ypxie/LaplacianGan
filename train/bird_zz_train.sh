
D=256
name='zz_mmgan_plain_gl_disc_continue_ncric'

## if you pretrained from an outside model 
root=../Models
folder=${name}_birds_${D}   
mkdir -p ${root}/${folder}

# echo 'Copy pretrained models'
# cp -v ${root}/zz_mmgan_noupsample_revisedisc_birds_64/G_epoch${epoch}.pth ${root}/${name}_birds_${D}/G_epoch${epoch}.pth
# cp -v ${root}/zz_mmgan_noupsample_revisedisc_birds_64/D_epoch${epoch}.pth ${root}/${name}_birds_${D}/D_epoch${epoch}.pth

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} --reuse_weights --load_from_epoch 100 --g_lr 0.0001 --d_lr 0.0001 | tee ${root}/${folder}/log.txt

#CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 16 --imsize ${D} --model_name ${name} | tee ${root}/${folder}/log.txt
