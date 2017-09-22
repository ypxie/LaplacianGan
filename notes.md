
## Small Model
1. 64*64 with 0.006 learning rate, batch size 16.  With skip connection in generator.
2. 64*64 with 0.002 lr, batch_size 16, without skil connection in gen.

## Large Model
## Ongoing testing
- large_shared_skip: testing shared disc + one more 1x1 convolution in image disc.

## Question need to answer:

- Small batch size?
    - Small batch size totally fails. The mode clapse is obvious. Batch size matters
- without large multimodal path?
- Balance or unbalanced
- Layer normalization+reLU  (LeakyreLU) is not working .... why? Wrong implementation?
- In DiscClassifier. Do we need an extra 1x1 convolutional to embed image feature.
    - Now the text is corresponded with image, because of removed unsample or 1x1 convolution?
