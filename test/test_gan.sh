# name='zz_mmgan_plain_gl_disc'
# gen='origin'

name='zz_mmgan_plain_gl_disc_ncric_single_256'
gen='single_256'

# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 200 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 300 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 400 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 560 --which_gen ${gen} --which_disc ${disc} 


#name='large_plan'
#gen='origin'
#disc='origin_global_local'
#python test_gan.py --device_id 3 --batch_size 32  --imsize 256 --model_name ${name}  --load_from_epoch 237 --which_gen ${gen} --which_disc ${disc} #

CUDA_VISIBLE_DEVICES=0   python test_gan.py --device_id 0 --train_mode --batch_size 1   --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} 