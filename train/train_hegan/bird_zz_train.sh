
D=256
name='zz_mmgan_plain_gl_disc'
# epoch=580

## if you pretrained from an outside model 
root=/home/zizhao/work/LaplacianGan/Models
folder=${name}_birds_${D}
mkdir -p ${root}/${folder}
# echo 'Copy pretrained models'
# cp -v ${root}/zz_mmgan_noupsample_revisedisc_birds_64/G_epoch${epoch}.pth ${root}/${name}_birds_${D}/G_epoch${epoch}.pth
# cp -v ${root}/zz_mmgan_noupsample_revisedisc_birds_64/D_epoch${epoch}.pth ${root}/${name}_birds_${D}/D_epoch${epoch}.pth

CUDA_VISIBLE_DEVICES=${device} python bird_train.py --batch_size 8 --imsize ${D} --model_name ${name} --which_disc origin | tee ${root}/${folder}/log.txt
