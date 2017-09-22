
name='large_shared_skip'

#root=/home/zizhao/work/LaplacianGan/Models
#mkdir -p ${root}/${name}_birds_64

#cp -v ${root}/zz_mmgan_birds_64/G_epoch300.pth ${root}/${name}_birds_64/G_epoch300.pth
#cp -v ${root}/zz_mmgan_birds_64/D_epoch300.pth ${root}/${name}_birds_64/D_epoch300.pth

#python bird_zz_train.py --device_id 2 --batch_size 12 --imsize 128 --model_name ${name} --reuse_weigths --load_from_epoch 2 --which_gen ${name} --save_freq 2
python bird_zz_train.py --device_id 2 --batch_size 12 --imsize 128 --model_name ${name} --reuse_weigths  --load_from_epoch 4 --which_gen ${name} --which_disc ${name} --save_freq 1
