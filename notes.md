
## Small Model
1. 64*64 with 0.006 learning rate, batch size 16.  With skip connection in generator.
2. 64*64 with 0.002 lr, batch_size 16, without skil connection in gen.

## Large Model
## Ongoing testing
- large_shared_skip: testing shared disc + one more 1x1 convolution in image disc.
- no_upsampling skip conection: **I do not think adding skip connect is necessary anymore**. The results are also good
- MultiStage architecture: Results seems promising. Stil training.
- 256 early version: Although the style is not corresponded. But I think the major issuse here is the smaller 256 does not have good quality. So 256 can not. 

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