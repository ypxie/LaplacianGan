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



class Generator(nn.Module):
    def __init__(self, input_size, sent_dim,  noise_dim, num_chan, 
                 emb_dim, hid_dim, norm = 'ln',branch=True, small_output=True):
        super(Generator, self).__init__()
        self.__dict__.update(locals())
        self.register_buffer('device_id', torch.zeros(1))
        self.condEmbedding = condEmbedding(sent_dim, emb_dim)
        # We need to know how many layers we will use at the beginning
        self.s = 64 # we fix it to 64
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        activ = genAct()

        node1_0 = sentConv(emb_dim+noise_dim, self.s16, self.s16, self.hid_dim*8, activ)
        _layers = conv_norm(self.hid_dim*8,  self.hid_dim*2, norm,  activ, 0, False,True,  1, 0)
        _layers += conv_norm(self.hid_dim*2,  self.hid_dim*2, norm, activ, 0, False,True,  3, 1)
        _layers += conv_norm(self.hid_dim*2,  self.hid_dim*8, norm, activ, 0, False,False,  3, 1)
        node1_1 = nn.Sequential(*_layers)
        self.node1  = resConn(node1_0, node1_1, activ)

        _layers = [nn.Upsample((self.s8, self.s8),mode='nearest')]
        _layers += conv_norm(self.hid_dim*8,  self.hid_dim*4, norm, activ, 0, False,False,  3, 1)
        node2_0 = nn.Sequential(*_layers)
        
        _layers = conv_norm(self.hid_dim*4,  self.hid_dim*1, norm,  activ, 0, False,True,  1, 0)
        _layers += conv_norm(self.hid_dim*1,  self.hid_dim*1, norm,  activ, 0, False,True, 3, 1)
        _layers += conv_norm(self.hid_dim*1,  self.hid_dim*4, norm,  activ, 0, False,False, 3, 1)
        node2_1 = nn.Sequential(*_layers)
        self.node2  = resConn(node2_0, node2_1, activ)

        _layers  = [nn.Upsample((self.s4, self.s4), mode='nearest')]
        _layers += conv_norm(self.hid_dim*4,  self.hid_dim*2, norm,  activ, 0, False,True, 3, 1)
        _layers += [nn.Upsample((self.s2, self.s2), mode='nearest')]
        _layers += conv_norm(self.hid_dim*2,  self.hid_dim*2, norm,  activ, 0, False,True, 3, 1)
        _layers += [nn.Upsample((self.s, self.s), mode='nearest')]
        _layers += conv_norm(self.hid_dim*2,  self.hid_dim,   norm,  activ, 0, False,True, 3, 1)   
        self.node_64 = nn.Sequential(*_layers)

        if branch or small_output: 
            _layers = [nn.Conv2d(self.hid_dim,  self.num_chan, 
                        kernel_size = 3, padding=1)]
            _layers += [nn.Tanh()]
            self.out_64 = nn.Sequential(*_layers)

        if not small_output: # means we need 256 outputs
            if branch:
                self.img_enc = Image64to16(input_size, num_chan, 
                                           self.hid_dim, self.hid_dim)
                self.text_enc = condEmbedding(sent_dim, emb_dim)
                
                com_dim = self.emb_dim + self.hid_dim
                
                _layers = conv_norm(com_dim,  self.hid_dim*2, norm,  activ, 0, False,True, 3, 1)

                _layers += [nn.Upsample((32, 32), mode='nearest')]
                _layers += conv_norm(self.hid_dim*2,  self.hid_dim*2, norm,  activ, 0, False,True, 3, 1)

                _layers += [nn.Upsample((64, 64), mode='nearest')]
                _layers += conv_norm(self.hid_dim*2,  self.hid_dim, norm,  activ, 0, False,True, 3, 1)

                self.comp_64 = nn.Sequential(*_layers)

            else:
                pass

            _layers = [nn.Upsample((128, 128), mode='nearest')]
            _layers += conv_norm(self.hid_dim,  self.hid_dim//2, norm,  activ, 0, False,True, 3, 1)

            _layers += [nn.Upsample((256, 256), mode='nearest')]
            _layers += conv_norm(self.hid_dim//2,  self.hid_dim//4, 'no',  activ, 0, False,True, 3, 1)

            _layers += [nn.Conv2d(self.hid_dim//4,  self.num_chan, kernel_size = 3, padding=1)]
            _layers += [nn.Tanh()]
            self.out_256 =  nn.Sequential(*_layers)

        self.apply(weights_init)

    def forward(self, sent_embeddings, z=None):
        out_dict = OrderedDict()
        sent_random, kl_loss  = self.condEmbedding(sent_embeddings)

        com_inp = torch.cat([sent_random, z], dim=1)
        #com_inp = inputs
        #node1_0 = self.node1_0(com_inp).view(-1,self.hid_dim*8, self.s16, self.s16)
        #node1_1 = self.node1_1(node1_0)
        #node1 = F.relu(node1_0 + node1_1)
        node1 = self.node1(com_inp)
        #node2_0 = self.node2_0(node1)
        #node2_1 = self.node2_1(node2_0)
        #node2  = F.relu(node2_0 + node2_1)
        node2 = self.node2(node1)
        node_64 = self.node_64(node2)

        if self.branch or self.small_output:
            output_64 =  self.out_64(node_64)
            out_dict['output_64']    = output_64
        
        if not self.small_output:
            if self.branch:
                img_enc = self.img_enc(output_64)
                text_enc, second_kl = self.text_enc(sent_embeddings)
                kl_loss += second_kl
                com_inp = cat_vec_conv(text_enc, img_enc)
                comp_64 = self.comp_64(com_inp)
            else:
                comp_64 = node_64
                
            output_256 = self.out_256(comp_64)
            out_dict['output_256']  = output_256
        #branch_128 = self.branch_128(node_128)

        return out_dict, kl_loss

class Image64to16(torch.nn.Module):
    '''
       This module encode image to 16*16 feat maps
    '''
    def __init__(self, input_size, num_chan, hid_dim, out_dim, norm = 'ln'):
        super(Image64to16, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.__dict__.update(locals())
        feat_size = input_size
        activ = genAct()

        _layers = conv_norm(num_chan, hid_dim,   norm,  activ, 0, False,True, 3, 1, 1)
        _layers += conv_norm(hid_dim, hid_dim*2, norm,  activ, 0, False,True, 3, 1, 2)
        _layers += conv_norm(hid_dim*2, out_dim, norm,  activ, 0, False,True, 3, 1, 2)

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
        activ = discAct()

        _layers = conv_norm(num_chan, hid_dim,   norm,  activ, 0, False,True, 3, 1, 2)
        _layers += conv_norm(hid_dim, hid_dim*2, norm,  activ, 0, False,True, 3, 1, 2)
        _layers += conv_norm(hid_dim*2, hid_dim*4, norm,  activ, 0, False,True, 3, 1, 2)

        if input_size == 64:
            self.content_node = nn.Sequential(*_layers)
        elif input_size == 256:
            _layers += conv_norm(hid_dim*4, hid_dim*2, norm,  activ, 0, False,True, 3, 1, 2)
            _layers += conv_norm(hid_dim*2, hid_dim*4, norm,  activ, 0, False,True, 3, 1, 2)
            self.content_node = nn.Sequential(*_layers)

        self.node1_0 = conv_norm(hid_dim*4, hid_dim*8, norm,  activ, 0, True, False, 3, 1, 2)
        
        _layers = conv_norm(hid_dim*8, hid_dim*2,   norm,  activ, 1, False,True, 1,0,1)
        _layers += conv_norm(hid_dim*2, hid_dim*8,   norm,  activ, 1, False,False, 1,0,1)
        
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
        
        activ =  discAct()
        _layers = [nn.Linear(sent_dim, emb_dim)]
        _layers += [discAct()]
        self.context_emb_pipe = nn.Sequential(*_layers)

        self.img_encoder = ImageEncoder(input_size, num_chan, hid_dim, enc_dim, norm=norm)
        
        # it is ugly to hard written,but to share weights between them.
        self.img_encoder_64 = ImageEncoder(64, num_chan, hid_dim, enc_dim, norm=norm)
        enc_dim = self.hid_dim * 8
        composed_dim = enc_dim + emb_dim
        
        _layers  = conv_norm(composed_dim, hid_dim*4,   norm,  activ, 0, False,True, 1,0,1)
        _layers += conv_norm(hid_dim*4, hid_dim*2,   norm,  activ, 0, False,True, 1,0,1)
        _layers += [nn.Conv2d(self.hid_dim*2, 1, kernel_size = 4, padding = 0, bias=True)]
        self.final_stage = nn.Sequential(*_layers)

        _layers  = conv_norm(enc_dim, hid_dim*4,   norm,  activ, 1, False,True, 1,0,1)
        _layers += conv_norm(hid_dim*4, hid_dim*2,   norm,  activ, 0, False,True, 1,0,1)
        _layers += [nn.Conv2d(hid_dim*2, 1, kernel_size = 4, padding = 0, bias=True)]
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
        
        compose_input = cat_vec_conv(sent_code, img_code)
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
