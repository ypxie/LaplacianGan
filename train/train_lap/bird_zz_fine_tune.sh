
D=256
name='zz_mmgan_pretrained64_plain'
epoch=580

## if you pretrained from an outside model 
# root=/home/zizhao/work/LaplacianGan/Models
# mkdir -p ${root}/${name}_birds_256
# echo 'Copy pretrained models'
# cp -v ${root}/zz_mmgan_noupsample_revisedisc_birds_64/G_epoch${epoch}.pth ${root}/${name}_birds_${D}/G_epoch${epoch}.pth
# cp -v ${root}/zz_mmgan_noupsample_revisedisc_birds_64/D_epoch${epoch}.pth ${root}/${name}_birds_${D}/D_epoch${epoch}.pth

CUDA_VISIBLE_DEVICES=${device} python bird_zz_train.py --batch_size 7 --imsize ${D} --model_name ${name} --reuse_weigths --load_from_epoch ${epoch} --which_gen origin_no_skip 
