
D=256
name='zz_mmgan_plain_gl_disc_baldg'
epoch=380

## if you pretrained from an outside model 
root=/home/zizhao/work/LaplacianGan/Models
folder=${name}_flowers_${D}
mkdir -p ${root}/${folder}

CUDA_VISIBLE_DEVICES=${device} python train.py --dataset flowers --batch_size 16 --imsize ${D} --model_name ${name} | tee ${root}/${folder}/log.txt
