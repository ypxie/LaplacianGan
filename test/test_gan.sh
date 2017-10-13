name='zz_mmgan_plain_gl_disc'
gen='origin'
disc='origin'
save_spec='train_bs_1'

CUDA_VISIBLE_DEVICES=1 python test_gan.py --device_id 0 --train_mode --batch_size 1 --save_spec ${save_spec}  --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} --which_disc ${disc} 
CUDA_VISIBLE_DEVICES=1 python test_gan.py --device_id 0 --train_mode --batch_size 4 --save_spec ${save_spec}  --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} --which_disc ${disc} 
CUDA_VISIBLE_DEVICES=1 python test_gan.py --device_id 0 --train_mode --batch_size 16 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} --which_disc ${disc} 

# The following for eval mode
save_spec='eval_bs_1'
CUDA_VISIBLE_DEVICES=1 python test_gan.py --device_id 0 --test_sample_num 11 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} --which_disc ${disc} 

# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 200 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 300 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 400 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} --which_disc ${disc} 


# name='zz_mmgan_plain_gl_disc'
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 200 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 300 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 400 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 500 --which_gen ${gen} --which_disc ${disc} 
# python test_gan.py --device_id 3 --batch_size 1 --save_spec ${save_spec} --imsize 256 --model_name ${name}  --load_from_epoch 560 --which_gen ${gen} --which_disc ${disc} 






#name='large_plan'
#gen='origin'
#disc='origin_global_local'
#python test_gan.py --device_id 3 --batch_size 32  --imsize 256 --model_name ${name}  --load_from_epoch 237 --which_gen ${gen} --which_disc ${disc} #

#rm -r /data/data2/yuanpu/Results
#cp -r ../Data/Results  /data/data2/yuanpu/
