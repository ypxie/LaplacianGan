import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict
from ..proj_utils.model_utils import *

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

class Generator(nn.Module):
    def __init__(self, input_size, sent_dim,  noise_dim, num_chan, 
                 emb_dim, hid_dim, norm = 'ln',branch=True, small_output=True):
        super(Generator, self).__init__()
        self.__dict__.update(locals())
        self.register_buffer('device_id', torch.zeros(1))
        self.condEmbedding = condEmbedding(sent_dim, noise_dim)

        # We need to know how many layers we will use at the beginning
        self.s = 64 # we fix it to 64
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        
        _layers = [nn.Linear(sent_dim, emb_dim)]
        _layers += [nn.Tanh()]
        self.text_enc_1 = nn.Sequential(*_layers)
        
        _layers = [nn.Linear(self.emb_dim + self.noise_dim, 
                   self.s16*self.s16 * hid_dim*8)]
        _layers += [nn.BatchNorm1d(self.s16*self.s16 
                    * self.hid_dim*8)]
        self.node1_0 = nn.Sequential(*_layers)
        
        _layers = [nn.Conv2d(self.hid_dim*8,  self.hid_dim*2, 
                    kernel_size = 1)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [genAct()]
        _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim*2, 
                    kernel_size = 3, padding=1)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [genAct()]
        _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim*8, 
                    kernel_size = 3, padding=1)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 8)]
        self.node1_1 = nn.Sequential(*_layers)
        
        _layers = [nn.Upsample((self.s8, self.s8),mode='nearest')]
        _layers += [nn.Conv2d(self.hid_dim*8,  self.hid_dim*4, 
                    kernel_size = 3, padding=1)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 4)]
        self.node2_0 = nn.Sequential(*_layers)
        
        _layers = [nn.Conv2d(self.hid_dim*4, self.hid_dim*1, 
                    kernel_size = 1)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 1)]
        _layers += [genAct()]
        _layers += [nn.Conv2d(self.hid_dim*1,  self.hid_dim*1, 
                    kernel_size = 3, padding=1)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 1 )]
        _layers += [genAct()]
        _layers += [nn.Conv2d(self.hid_dim*1,  self.hid_dim*4, 
                    kernel_size = 3, padding=1)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 4)]
        self.node2_1 = nn.Sequential(*_layers)

        _layers = [nn.Upsample((self.s4, self.s4), mode='nearest')] # 16
        _layers += [nn.Conv2d(self.hid_dim*4,  self.hid_dim*2, 
                    kernel_size = 3, padding=1)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [genAct()]
        self.node_16 = nn.Sequential(*_layers)

        if branch or small_output:
            _layers = [nn.Upsample((self.s2, self.s2), mode='nearest')] # 32
            _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim*2, 
                        kernel_size = 3, padding=1)]
            _layers += [nn.BatchNorm2d(self.hid_dim *2)]
            _layers += [genAct()]

            _layers += [nn.Upsample((self.s, self.s), mode='nearest')] # 64
            _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim, 
                        kernel_size = 3, padding=1)] 
            _layers += [nn.BatchNorm2d(self.hid_dim )]    
            _layers += [genAct()]          
            #self.node_64 = nn.Sequential(*_layers)

            _layers += [nn.Conv2d(self.hid_dim,  self.num_chan, 
                        kernel_size = 3, padding=1)]
            _layers += [nn.Tanh()]
            self.out_64 = nn.Sequential(*_layers)

        if not small_output: # means we need 256 outputs
            
            _layers = [nn.Linear(sent_dim, emb_dim)]
            _layers += [nn.Tanh()]
            self.text_enc_2 = nn.Sequential(*_layers)
            
            #self.condEmbedding = condEmbedding(sent_dim, noise_dim)


            com_dim = self.emb_dim + self.hid_dim * 2
            
            _layers = [nn.Conv2d(com_dim,  self.hid_dim*2, 
                kernel_size = 3, padding=1)]
            _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
            _layers += [genAct()]

            _layers += [nn.Upsample((32, 32), mode='nearest')]
            _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim*2, 
                        kernel_size = 3, padding=1)]
            _layers += [nn.BatchNorm2d(self.hid_dim*2 )]
            _layers += [genAct()]

            _layers += [nn.Upsample((64, 64), mode='nearest')]
            _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim, 
                        kernel_size = 3, padding=1)]
            _layers += [nn.BatchNorm2d(self.hid_dim)]
            _layers += [genAct()]
            
            _layers += [nn.Upsample((128, 128), mode='nearest')]
            _layers += [nn.Conv2d(self.hid_dim,  self.hid_dim//2, 
                        kernel_size = 3, padding=1)]
            _layers += [nn.BatchNorm2d(self.hid_dim//2 )]
            _layers += [genAct()]

            _layers += [nn.Upsample((256, 256), mode='nearest')]
            _layers += [nn.Conv2d(self.hid_dim//2,  self.hid_dim//4, 
                        kernel_size = 3, padding=1)]
            _layers += [genAct()]
            _layers += [nn.Conv2d(self.hid_dim//4,  self.num_chan, 
                        kernel_size = 3, padding=1)]
            _layers += [nn.Tanh()]
            self.out_256 =  nn.Sequential(*_layers)

        self.apply(weights_init)

    def forward(self, sent_embeddings, z=None):
        # we don't use z
        out_dict = OrderedDict()
        sent_random  = self.text_enc_1(sent_embeddings)
        z, kl_loss_ = self.condEmbedding(sent_embeddings)
        com_inp = torch.cat([sent_random, z], dim=1)
        #com_inp = inputs
        node1_0 = self.node1_0(com_inp).view(
                  -1,self.hid_dim*8, self.s16, self.s16)
        node1_1 = self.node1_1(node1_0)
        node1 = F.relu(node1_0 + node1_1)

        node2_0 = self.node2_0(node1)
        node2_1 = self.node2_1(node2_0)
        node2  = F.relu(node2_0 + node2_1)
        node_16 = self.node_16(node2)

        if self.branch or self.small_output:
            output_64 =  self.out_64(node_16)
            out_dict['output_64']    = output_64

        if not self.small_output:
            img_enc = node_16
            text_enc  = self.text_enc_2(sent_embeddings)
            #kl_loss_ += second_kl

            b, c = text_enc.size()
            row, col = img_enc.size()[2::]
            text_enc = text_enc.unsqueeze(-1).unsqueeze(-1)
            text_enc = text_enc.expand(b, c, row, col )
            com_inp = torch.cat([img_enc, text_enc], 1)
            
            output_256 = self.out_256(com_inp)
            out_dict['output_256']  = output_256
        #branch_128 = self.branch_128(node_128)

        return out_dict, kl_loss_

class Image64to16(torch.nn.Module):
    '''
       This module encode image to 16*16 feat maps
    '''
    def __init__(self, input_size, num_chan, hid_dim, out_dim, norm = 'ln'):
        super(Image64to16, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.__dict__.update(locals())
        feat_size = input_size

        _layers = [nn.Conv2d(num_chan,  self.hid_dim, 
                    kernel_size = 3, stride=1, padding=(1,1))]
        _layers += [genAct()]

        _layers += [nn.Conv2d(self.hid_dim,  self.hid_dim*2, 
                    kernel_size = 3, stride=2, padding=(1,1))]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [genAct()]

        _layers += [nn.Conv2d(self.hid_dim*2,  out_dim, 
                    kernel_size = 3, stride=2, padding=(1,1))]
        _layers += [nn.BatchNorm2d(out_dim)]
        _layers += [genAct()]
        
        self.node = nn.Sequential(*_layers)

    def forward(self, inputs):
        # inputs (B, C, H, W), must be dividable by 32
        # return (B, C, 16, 16)
        return self.node(inputs)

# DCGAN discriminator (using somewhat the reverse of the generator)
# Removed Batch Norm we can't backward on the gradients with BatchNorm2d

class ImageEncoder(torch.nn.Module):
    '''
       This module encode image to 4*4 feat maps
    '''
    def __init__(self, input_size, num_chan, hid_dim, out_dim, norm = 'ln'):
        super(ImageEncoder, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.__dict__.update(locals())
        feat_size = input_size

        _layers = [nn.Conv2d(num_chan,  self.hid_dim, 
                    kernel_size = 3, stride=2, padding=(1,1))]
        _layers += [discAct()]

        _layers += [nn.Conv2d(self.hid_dim,  self.hid_dim*2, 
                    kernel_size = 3, stride=2, padding=(1,1))]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [discAct()]

        _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim*4, 
                    kernel_size = 3, stride=2, padding=(1,1))]
        _layers += [nn.BatchNorm2d(self.hid_dim * 4)]
        _layers += [discAct()]

        if input_size == 64:
            self.content_node = nn.Sequential(*_layers)
        elif input_size == 256:

            _layers += [nn.Conv2d(self.hid_dim * 4,  self.hid_dim*2, 
                        kernel_size = 3, stride=2, padding=(1,1))]
            _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
            _layers += [discAct()]

            _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim*4, 
                        kernel_size = 3, stride=2, padding=(1,1))]
            _layers += [nn.BatchNorm2d(self.hid_dim * 4)]
            _layers += [discAct()]
            self.content_node = nn.Sequential(*_layers)

        _layers = [nn.Conv2d(self.hid_dim*4,  self.hid_dim*8, 
                    kernel_size = 3, stride=2, padding=(1,1))]
        _layers += [nn.BatchNorm2d(self.hid_dim * 8)]
        self.node1_0 = nn.Sequential(*_layers)

        _layers = [nn.Conv2d(self.hid_dim*8,  self.hid_dim*2, 
                    kernel_size = 1, stride=1, padding=0)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [discAct()]
        _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim*2, 
                    kernel_size = 1, stride=1, padding=0)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [discAct()]
        _layers += [nn.Conv2d(self.hid_dim*2,  self.hid_dim*8, 
                    kernel_size = 1, stride=1, padding=0)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 8)]
        self.node1_1 = nn.Sequential(*_layers)

    def forward(self, inputs):
        # inputs (B, C, H, W), must be dividable by 32
        # return (B, C, 4, 4)
        # node1_0 for feature content loss

        content_code = self.content_node(inputs)
        node1_0 = self.node1_0(content_code)
        node1_1 = self.node1_1(node1_0)
        node1 = F.leaky_relu(node1_0 + node1_1, 0.2)
        return node1, content_code


class Discriminator(torch.nn.Module):
    '''
    enc_dim: Reduce images inputs to (B, enc_dim, H, W)
    emb_dim: The sentence embedding dimension.
    '''

    def __init__(self, input_size, num_chan,  hid_dim, 
                sent_dim,  enc_dim, emb_dim, norm='ln'):
        super(Discriminator, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.__dict__.update(locals())
        
        _layers = [nn.Linear(sent_dim, emb_dim)]
        #_layers += [nn.BatchNorm1d(emb_dim)]
        _layers += [discAct()]
        _layers += [nn.Linear(emb_dim, emb_dim)]
        #_layers += [nn.BatchNorm1d(emb_dim*4*4)]
        _layers += [discAct()]
        self.context_emb_pipe = nn.Sequential(*_layers)

        self.img_encoder = ImageEncoder(input_size, num_chan, hid_dim, enc_dim, norm=norm)
        
        # it is ugly to hard written,but to share weights between them.
        self.img_encoder_64 = ImageEncoder(64, num_chan, hid_dim, enc_dim, norm=norm)
        enc_dim = self.hid_dim * 8
        composed_dim = enc_dim + emb_dim

        _layers = [nn.Conv2d(composed_dim, self.hid_dim*4, 
                    kernel_size = 1, padding = 0, bias=True)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 4)]
        _layers += [discAct()]
        _layers += [nn.Conv2d(self.hid_dim*4, self.hid_dim*2, 
                    kernel_size = 1, padding = 0, bias=True)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [discAct()]
        _layers += [nn.Conv2d(self.hid_dim*2, 1, 
                    kernel_size = 4, padding = 0, bias=True)]
        self.final_stage = nn.Sequential(*_layers)

        _layers = [nn.Conv2d(enc_dim, self.hid_dim*2, 
                    kernel_size = 1, padding = 0, bias=True)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [discAct()]
        _layers += [nn.Conv2d(self.hid_dim*2, self.hid_dim*2, 
                    kernel_size = 1, padding = 0, bias=True)]
        _layers += [nn.BatchNorm2d(self.hid_dim * 2)]
        _layers += [discAct()]
        _layers += [nn.Conv2d(self.hid_dim*2, 1, 
                    kernel_size = 4, padding = 0, bias=True)]
        self.img_disc = nn.Sequential(*_layers)
        
    def encode_img(self, images):
        img_size = images.size()[3]
        #print('images size ', images.size())
        if img_size == 64:
            img_code, content_code = self.img_encoder_64(images)
        else:    
            img_code, content_code = self.img_encoder(images)
        return img_code, content_code
    
    def forward(self, images, embdding):
        '''
        images: (B, C, H, W)
        embdding : (B, sent_dim)
        outptu: (B, 1)
        '''
        out_dict = OrderedDict()
        img_size = images.size()[3]

        img_code, content_code  = self.encode_img(images)
        sent_code = self.context_emb_pipe(embdding)
        
        sent_code =  sent_code.unsqueeze(-1).unsqueeze(-1)
        dst_shape = list(sent_code.size())
        #print(dst_shape, img_code.size())
        dst_shape[1] = self.emb_dim
        dst_shape[2] =  img_code.size()[2] 
        dst_shape[3] =  img_code.size()[3] 
        sent_code = sent_code.expand(dst_shape)
        #sent_code = sent_code.view(*dst_shape)

        compose_input = torch.cat([img_code, sent_code], dim=1)
        #print('compose_input size ', compose_input.size())
        
        output = self.final_stage(compose_input)
        img_disc = self.img_disc(img_code)

        #print('output shape is ', output.size())
        b, outdim = output.size()[0:2]
        b_i, out_dim_i = img_disc.size()[0:2]

        out_dict['pair_disc'] = output.view(b, outdim)
        out_dict['img_disc']  = img_disc.view(b_i, out_dim_i)
        out_dict['content_code']  = content_code
        return out_dict

## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)
