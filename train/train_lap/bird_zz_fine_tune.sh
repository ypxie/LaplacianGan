
name='zz_mmgan_ft_nocontent'

root=/home/zizhao/work/LaplacianGan/Models
mkdir -p ${root}/${name}_birds_64

cp -v ${root}/zz_mmgan_birds_64/G_epoch300.pth ${root}/${name}_birds_64/G_epoch300.pth
cp -v ${root}/zz_mmgan_birds_64/D_epoch300.pth ${root}/${name}_birds_64/D_epoch300.pth

python bird_zz_train.py --batch_size 16 --imsize 64 --model_name ${name} --reuse_weigths --load_from_epoch 300 --debug_mode
