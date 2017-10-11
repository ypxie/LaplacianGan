
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



# Model saved
- zz_mmgan_noupsample_revisedisc_birds_64: do not use any upsample skip connection, pair discriminator has an extra 1x1
- zz_mmgan_256_birds_256: has pretty resonable results, but the style is inaccurate


# To Do
- local disc 
- Use a buffer to save low scored samples up to 500, and take half of fake samples from thoese buffer. at each iteration, update buffer. looks like buffer reply. 
- using 4096X16X16 memory to help the network learning.


# inception score
- StackGAN (reported) ha mean 3.89(3.70) std: 0.05 (0.04)
- ~~use balanced disc has mean: 3.41 std: 0.04~~
- ~~use unbalanced disc (ncric=5) has mean: 3.53 std: 0.03~~


- ~~zz_mmgan_plain_gl_disc_birds_256_G_epoch_400.h5: 3.43 0.04~~
- ~~zz_mmgan_plain_gl_disc_birds_256_G_epoch_300.h5: 3.44 0.05~~
- zz_mmgan_plain_gl_disc_birds_256_G_epoch_500.h5 mean: 4.01 std: 0.04
- zz_mmgan_plain_gl_disc_birds_256_G_epoch_560.h5 mean: 4.0 std 0.03
- zz_mmgan_plain_gl_disc_birds_256_G_epoch_300.h5 mean: 3.96 std: 0.02
- zz_mmgan_plain_gl_disc_birds_256_G_epoch_400.h5 mean: 3.99 std: 0.02
- eval_bs_1testing_num_11/ mean: 3.99 std: 0.05