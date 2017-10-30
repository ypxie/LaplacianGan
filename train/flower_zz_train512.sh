
D=512
name='zz_mmgan_plain_local_512_ncit30_50decay_super3'

## if you pretrained from an outside model 
root=../Models
folder=${name}_flowers_${D}   
mkdir -p ${root}/${folder}

cp -v ../Models/zz_mmgan_plain_gl_disc_baldg2_flowers_256/G_epoch580.pth ${root}/${folder}/G_epoch_256_init.pth

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset flowers --batch_size 8 --imsize ${D} --model_name ${name} --which_gen super2 --which_disc super --ncritic_epoch_range 30 --epoch_decay 50 --maxepoch 600 | tee ${root}/${folder}/log.txt

