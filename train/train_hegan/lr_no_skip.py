
#name='large_shared_skip'
name='large_lr_no_skip'
gen='origin'
disc='origin_global_local'
python bird_train.py --device_id 1 --g_lr 0.0005 --d_lr 0.0005 --emb_interp --batch_size 16 --num_emb 4 --imsize 256 --model_name ${name} --load_from_epoch 3 --which_gen ${gen} --which_disc ${disc} --save_freq 1

