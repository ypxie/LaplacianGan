
D=512
name='zz_mmgan_plain_local_512_ncit0_20decay_res3_img2_nopair'

## if you pretrained from an outside model 
root=../Models
folder=${name}_birds_${D}   
mkdir -p ${root}/${folder}

cp -v ../Models/zz_mmgan_plain_gl_disc_birds_256/G_epoch500.pth ${root}/${folder}/G_epoch_256_init.pth

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset birds --batch_size 12 --imsize ${D} --model_name ${name} --which_gen super_l1loss --which_disc super --ncritic_epoch_range 0 --epoch_decay 20 --maxepoch 100 --save_freq 1 | tee ${root}/${folder}/log.txt

