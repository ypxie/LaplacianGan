
#name='large_shared_skip'
name='large_no_skip'
python bird_zz_train.py --device_id 2 --emb_interp --batch_size 16 --num_emb 4 --imsize 128 --model_name ${name} --load_from_epoch 3 --which_gen ${name} --which_disc ${name} --save_freq 1

#name='det_large_shared_skip'
#python bird_zz_train.py --device_id 2 --g_lr 0.0005 --d_lr 0.0005 --batch_size 8 --num_emb 1 --imsize 256 --model_name ${name} --reuse_weights  --load_from_epoch 0 --which_gen ${name} --which_disc ${name} --save_freq 1