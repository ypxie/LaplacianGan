
name='large_shared_skip'

#root=/home/zizhao/work/LaplacianGan/Models
#mkdir -p ${root}/${name}_birds_64

#cp -v ${root}/zz_mmgan_birds_64/G_epoch300.pth ${root}/${name}_birds_64/G_epoch300.pth
#cp -v ${root}/zz_mmgan_birds_64/D_epoch300.pth ${root}/${name}_birds_64/D_epoch300.pth

python bird_zz_train.py --device_id 2 --batch_size 8 --imsize 256 --model_name ${name} --load_from_epoch 300 --which_gen ${name}
