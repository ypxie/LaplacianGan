
#name='large_shared_skip'
name='large_plan'
gen='origin'
disc='origin_global_local'

#python bird_train.py --device_id 2 --batch_size 16 --num_emb 4 --imsize 256 --model_name ${name} --reuse_weights --load_from_epoch 138 --which_gen ${gen} --which_disc ${disc} --save_freq 3
python train.py --device_id 0 --g_lr 0.0001 --d_lr 0.0001 --batch_size 16 --num_emb 4 --imsize 256 --model_name ${name} --reuse_weights --load_from_epoch 237 --which_gen ${gen} --which_disc ${disc} --save_freq 3

#python train.py --device_id 1 --g_lr 0.0005 --d_lr 0.0005 --batch_size 16 --num_emb 4 --imsize 256 --model_name ${name} --load_from_epoch 3 --which_gen ${gen} --which_disc ${disc} --save_freq 1

#name='det_large_shared_skip'
#python bird_zz_train.py --device_id 2 --g_lr 0.0005 --d_lr 0.0005 --batch_size 8 --num_emb 1 --imsize 256 --model_name ${name} --reuse_weights  --load_from_epoch 0 --which_gen ${name} --which_disc ${name} --save_freq 1
