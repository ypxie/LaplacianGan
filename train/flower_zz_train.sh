
D=256
name='zz_mmgan_plain_gl_disc'

## if you pretrained from an outside model 
root=../Models
folder=${name}_flowers_${D}
mkdir -p ${root}/${folder}

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset flowers --batch_size 16 --imsize ${D} --model_name ${name} | tee ${root}/${folder}/log.txt
