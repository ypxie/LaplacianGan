# name='zz_mmgan_plain_gl_disc'
# gen='origin'

name='zz_mmgan_plain_gl_disc_ncric_single_256'
gen='single_256'


CUDA_VISIBLE_DEVICES=0   python test_gan.py --device_id 0 --train_mode --batch_size 1   --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} 