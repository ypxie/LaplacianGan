import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict
from ..proj_utils.model_utils import *
import math


def branch_out2(in_dim, out_dim=3):
    _layers = [nn.ReflectionPad2d(1),
                nn.Conv2d(in_dim, out_dim, 
                kernel_size = 3, padding=0, bias=False)]    
    _layers += [nn.Tanh()]

    return nn.Sequential(*_layers)


def KL_loss(mu, log_sigma):
    loss = -log_sigma + .5 * (-1 + torch.exp(2. * log_sigma) + mu**2)
    loss = torch.mean(loss)
    return loss

def sample_encoded_context(mean, logsigma, kl_loss=False):
    
    # epsilon = tf.random_normal(tf.shape(mean))
    epsilon = to_device( torch.randn(mean.size()), mean, requires_grad=False) #tf.truncated_normal(tf.shape(mean))
    stddev  = torch.exp(logsigma)
    c = mean + stddev * epsilon

    kl_loss = KL_loss(mean, logsigma) if kl_loss else None
    return c, kl_loss

class resConn(nn.Module):
    def __init__(self, in_path, side_path, activ):
        super(resConn, self).__init__()
        self.in_path = in_path
        self.side_path = side_path
        self.activ = activ

    def forward(self, inputs):
        node_0 = self.in_path(inputs)
        node_1 = self.side_path(node_0)
        node2  = self.activ(node_0 + node_1)
        return node2


class condEmbedding(nn.Module):
    def __init__(self, noise_dim, emb_dim):
        super(condEmbedding, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.noise_dim = noise_dim
        self.emb_dim = emb_dim
        self.linear  = nn.Linear(noise_dim, emb_dim*2)
        
    def forward(self, inputs, kl_loss=True):
        '''
        inputs: (B, dim)
        return: mean (B, dim), logsigma (B, dim)
        '''
        out = F.leaky_relu( self.linear(inputs), 0.2 )
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]

        c, kl_loss = sample_encoded_context(mean, log_sigma, kl_loss)
        return c, kl_loss

def genAct():
    return nn.ReLU()
def discAct():
    return nn.LeakyReLU(0.2)

def conv_norm2(dim_in, dim_out, norm_layer, kernel_size=3, use_activation=True, use_bias=False):
     # nn.ReflectionPad2d(1) avoids use zero-padding in Conv2d
    seq = []
    if kernel_size != 1:
        seq += [nn.ReflectionPad2d(1)]

    seq += [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=0, bias=use_bias),
          norm_layer(dim_out)]
    
    if use_activation:
        seq += [nn.ReLU(True)]
    
    return nn.Sequential(*seq)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm, use_bias=False):
        super(ResnetBlock, self).__init__()
        norm_layer = getNormLayer(norm)

        seq = [conv_norm2(dim, dim, norm_layer, use_bias=use_bias), 
               conv_norm2(dim, dim, norm_layer, use_activation=False, use_bias=use_bias)]
        self.res_block = nn.Sequential(*seq)

    def forward(self, input):
        # TODO do we need to add activation? 
        # CycleGan regards this. I guess to prevent spase gradients
        
        return self.res_block(input) + input

class MultiModalBlock(nn.Module):
    def __init__(self, text_dim, img_dim, norm, use_bias=False, upsample_factor=3):
        super(MultiModalBlock, self).__init__()
        norm_layer = getNormLayer(norm)
        # upsampling 2^3 times
        seq = []
        cur_dim = text_dim
        for i in range(upsample_factor):
            seq += [nn.Upsample(scale_factor=2, mode='nearest')]
            seq += [conv_norm2(cur_dim, cur_dim//2, norm_layer)]
            cur_dim /= 2

        self.upsample_path = nn.Sequential(*seq)
        self.joint_path = nn.Sequential(*[
            conv_norm2(cur_dim+img_dim, img_dim, norm_layer, kernel_size=1)
        ])
    def forward(self, text, img ):
        upsampled_text = self.upsample_path(text)
        
        out = self.joint_path(torch.cat([img, upsampled_text],1))
        return out


class Generator(nn.Module):
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, norm='ln', output_size=256):
        super(Generator, self).__init__()
        self.__dict__.update(locals())

        self.register_buffer('device_id', torch.zeros(1))
        self.condEmbedding = condEmbedding(sent_dim, emb_dim)

        # We need to know how many layers we will use at the beginning
       
        norm_layer = getNormLayer(norm)

        self.vec_to_tensor = sentConv(emb_dim+noise_dim, 4, 4, self.hid_dim*8, None, False)
        
        '''user defefined'''
        self.output_size = output_size
        # 64, 128, or 256 version
        
        if output_size == 256:
            num_scales = [4, 8, 16, 32, 64, 128, 256]
        elif output_size == 64:
            num_scales = [4, 8, 16, 32, 64]
        elif output_size == 128:
            num_scales = [4, 8, 16, 32, 64, 128]

        print ('>> initialized a {} size generator with {} outputs'.format(output_size, log(output_size/64,2)))

        reduce_dim_at = [8, 64, 256] 
        side_output_at = [64, 128, 256] 
        text_upsampling_at = [4, 8, 16] 
        num_resblock = 1

        self.modules = OrderedDict()
        self.side_modules = OrderedDict()

        cur_dim = self.hid_dim*8
        for i in range(len(num_scales)):
            seq = []
            # unsampling
            if i != 0:
                seq += [nn.Upsample(scale_factor=2, mode='nearest')]
            # if need to reduce dimension
            if num_scales[i] in reduce_dim_at:
                seq += [conv_norm2(cur_dim, cur_dim//2, norm_layer)]
                cur_dim /= 2
            # print ('scale {} cur_dim {}'.format(num_scales[i], cur_dim))
            # add residual blocks
            for n in range(num_resblock):
                seq += [ResnetBlock(cur_dim, norm, use_bias=False)]
            # add main convolutional module
            setattr(self, 'scale_%d'%(num_scales[i]), nn.Sequential(*seq) )

            # add upsample module to concat with upper layers 
            if num_scales[i] in text_upsampling_at:
                setattr(self, 'upsample_%d'%(num_scales[i]), MultiModalBlock(cur_dim, cur_dim//2, norm))
            # configure side output module
            if num_scales[i] in side_output_at:
                setattr(self, 'tensor_to_img_%d'%(num_scales[i]), branch_out2(cur_dim))

        
        self.apply(weights_init)

    def forward(self, sent_embeddings, z=None):
        out_dict = OrderedDict()
        sent_random, kl_loss  = self.condEmbedding(sent_embeddings)
        text = torch.cat([sent_random, z], dim=1)

        x = self.vec_to_tensor(text)
        x_4 = self.scale_4(x)
        x_8 = self.scale_8(x_4)
        x_16 = self.scale_16(x_8)
        x_32 = self.scale_32(x_16)
        
        # skip 4x4 feature map to 32 and send to 64
        x_32_4 = self.upsample_4(x_4, x_32)
        x_64 = self.scale_64(x_32_4)
        out_dict['output_64'] = self.tensor_to_img_64(x_64)
        
        if self.output_size > 64:
            # skip 8x8 feature map to 64 and send to 128
            x_64_8 = self.upsample_8(x_8, x_64)
            x_128 = self.scale_128(x_64_8)
            out_dict['output_128'] = self.tensor_to_img_128(x_128)

        if self.output_size > 128:
            # skip 16x16 feature map to 128 and send to 256
            x_128_16 = self.upsample_16(x_16, x_128)
            out_256 = self.scale_256(x_128_16)

            out_dict['output_256'] = self.tensor_to_img_256(out_256)


        return out_dict, kl_loss

