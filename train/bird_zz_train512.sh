
D=512
name='zz_mmgan_plain_local_512_ncit30_50decay_super4'

## if you pretrained from an outside model 
root=../Models
folder=${name}_birds_${D}   
mkdir -p ${root}/${folder}

cp -v ../Models/zz_mmgan_plain_gl_disc_birds_256/G_epoch500.pth ${root}/${folder}/G_epoch_256_init.pth

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 12 --imsize ${D} --model_name ${name} --which_gen super2 --which_disc super --ncritic_epoch_range 20 --epoch_decay 20 --maxepoch 200 --save_freq 1 | tee ${root}/${folder}/log.txt

