
name='zz_mmgan_plain_gl_disc'
gen='origin'
disc='origin'

python test_gan.py --device_id 3 --batch_size 16  --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} --which_disc ${disc} 

name='zz_mmgan_plain_gl_disc_baldg'
python test_gan.py --device_id 3 --batch_size 16  --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} --which_disc ${disc} 

