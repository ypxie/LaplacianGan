
## Small Model
1. 64*64 with 0.006 learning rate, batch size 16.  With skip connection in generator.
2. 64*64 with 0.002 lr, batch_size 16, without skil connection in gen.

## Large Model
## Ongoing testing
- large_shared_skip: testing shared disc + one more 1x1 convolution in image disc.
- no_upsampling skip conection: **I do not think adding skip connect is necessary anymore**. The results are also good
- MultiStage architecture: Results seems promising. Stil training.
- 256 early version: Although the style is not corresponded. But I think the major issuse here is the smaller 256 does not have good quality. So 256 can not. 

- 256 large_skip_shared: with nun_emb = 1.
- 64 no_skip_connection-thin network: with num_emb = 1. 

## Question need to answer:

- Small batch size?
    - Small batch size totally fails. The mode clapse is obvious. Batch size matters
- without large multimodal path?
- Balance or unbalanced
- Layer normalization+reLU  (LeakyreLU) is not working .... why? Wrong implementation?
- In DiscClassifier. Do we need an extra 1x1 convolutional to embed image feature.
    - **Now the text is corresponded with image, because of removed unsample or 1x1 convolution?**



# Model savede
- zz_mmgan_noupsample_revisedisc_birds_64: do not use any upsample skip connection, pair discriminator has an extra 1x1
- zz_mmgan_256_birds_256: has pretty resonable results, but the style is inaccurate


# To Do
- local disc 
- Use a buffer to save low scored samples up to 500, and take half of fake samples from thoese buffer. at each iteration, update buffer. looks like buffer reply. 
- using 4096X16X16 memory to help the network learning.


# inception score
- StackGAN (reported) birds mean 3.89(3.70) std: 0.05 (0.04)
- StackGAN (reported) flowers mean: 3.16(3.20) std: 0.03(0.01)

## Birds (WRONG Results)
- ~~use balanced disc has mean: 3.41 std: 0.04~~
- ~~use unbalanced disc (ncric=5) has mean: 3.53 std: 0.03~~
- ~~zz_mmgan_plain_gl_disc_birds_256_G_epoch_400.h5: 3.43 0.04~~ # mistakenly use training mode
- ~~zz_mmgan_plain_gl_disc_birds_256_G_epoch_300.h5: 3.44 0.05~~

### use ncritnic for the first 100 epoches
- **zz_mmgan_plain_gl_disc_birds_256_G_epoch_500.h5** mean: 4.01 std: 0.04
    - scale: output_64 mean: 3.461111068725586 std:0.042458079755306244
    - scale: output_128 mean: 3.8552157878875732 std:0.03652472421526909
    - scale: output_256 mean: 4.0113630294799805 std:0.05708415433764458
     second time (with 512): 4.07792329788208 std:0.05174040421843529
- zz_mmgan_plain_gl_disc_birds_256_G_epoch_560.h5 mean: 4.0 std 0.03
- zz_mmgan_plain_gl_disc_birds_256_G_epoch_300.h5 mean: 3.96 std: 0.02
- zz_mmgan_plain_gl_disc_birds_256_G_epoch_400.h5 mean: 3.99 std: 0.02
- eval_bs_1testing_num_11/ mean: 3.99 std: 0.05 # evaluate 11 images per data
- zz_mmgan_plain_gl_disc_ncric_fulglo_256_birds_256_G_epoch_500 scale: output_256 mean: 3.970097780227661 std:0.04089314490556717 (bug free)

## **Yuanpu's (bug free)**
- gen_origin_disc_global_no_img_birds_[64, 128, 256]_G_epoch_501 scale: output_256 mean: 4.0969719886779785 std:0.04281013458967209
- gen_origin_disc_both_birds_[64, 128, 256]_G_epoch_405 scale: output_256 mean: 4.0880842208862305 std:0.04456903785467148
- **gen_origin_disc_global_no_img_birds_[64, 128, 256]_G_epoch_597 scale: output_256 mean: 4.270041465759277 std:0.046753715723752975**

#### Different supervision
- zz_mmgan_plain_gl_disc_ncric_single_256_birds_256_G_epoch_500 output_256 mean: 3.518810749053955 std:0.044894989579916
- zz_mmgan_plain_gl_disc_ncric_comb_64_256v2_birds_256_G_epoch_500 scale: output_256 mean: 4.135245323181152 std:0.03427153825759888
- zz_mmgan_plain_gl_disc_ncric_comb_128_256_birds_256_G_epoch_500 scale scale: output_256 mean: 3.987717390060425 std:0.042137689888477325
### all use ncritic 
- zz_mmgan_plain_gl_disc_continue_ncric_birds_256_G_epoch_400.h5: mean 3.97 std: 0.03
- zz_mmgan_plain_gl_disc_continue_ncric_birds_256_G_epoch_500.h5: mean: 3.97 std: 0.03
- zz_mmgan_plain_gl_disc_continue_ncric_birds_256_G_epoch_300.h5: mean: 3.86 std: 0.05

## Flower
### do not use ncritic
zz_mmgan_plain_gl_disc_baldg2_flowers_256_G_epoch_500_inception_score (11550 samples): mean: 3.40 std: 0.07
zz_mmgan_plain_gl_disc_baldg2_flowers_256_G_epoch_580_inception_score (11550 samples): mean: 3.45 std: 0.07 
zz_mmgan_plain_gl_disc_baldg2_flowers_256_G_epoch_580_inception_score (30000 samples): mean: 3.45 std: 0.07     
zz_mmgan_plain_gl_disc_ncric_flowers_256_G_epoch_500_inception_score : 3.3664 std 0.023 # not as good as expected
zz_mmgan_plain_gl_disc_ncric10_flowers_256_G_epoch_500
mean: 3.34143 std:0.0315

## how to generate coco training data
- go to process_data/get_captions_coco.py to get [image]\_captions.txt and captions.pickle
- go to process_data/get_embedding_coco.lua to compute embedding in txt files [image]\_captions.txt.
- go to process_data/prepare_coco.py to merge embeddings in t7 to pickle and fileinfo.pickle

## COCO
gen_origin_disc_origin_coco_[64]_G_epoch_195_inception_score.json
mean: 8.25843334197998, std: 0.08759678900241852,

gen_origin_disc_origin_coco_[64]_G_epoch_261_inception_score.j  son
mean: 8.308503150939941 std: 0.13465775549411774

gen_origin_disc_origin_coco_[64, 128]_G_epoch_99.h5
mena: 8.0430 std: 0.130

gen_origin_disc_origin_coco_[64, 128]_G_epoch_114.h5
output_128 mean : 7.636, std: 0.117
output_64 mean 7.79899, 0.132714241

## Currently I am testing (all old models are backup. Currently bug-free version)
# does bigger model help? does local help?
- bigmachine: coco, global loss , 64 and 128
- bigmachine: birds_no_img_loss. birds_disc_both
- devbox:     birds with vanilla 256 model. 



# Ideas
- Why vanilla 256 GAN is hard to train?
- Large variances, training instability, graident vanishing, low-high mapping is 
- 
