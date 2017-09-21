import math
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

class connectSide(nn.Module):
    def __init__(self, side_in, side_out, hid_in, sent_in,out_dim, norm, activ, repeat= 0):
        super(connectSide, self).__init__()
        self.__dict__.update(locals())

        _layers = []
        _layers += [nn.Conv2d(side_in, side_out, kernel_size = 3, padding=1, bias=True)]
        _layers += [getNormLayer(norm)(side_out )]
        _layers += [activ]
        
        self.side_trans = nn.Sequential(*_layers)

        _layers = [nn.Conv2d(side_out + sent_in + hid_in,  out_dim,
                    kernel_size = 3, padding=1, bias=True)]
        _layers += [getNormLayer(norm)(out_dim )]
        _layers += [activ]

        for _ in range(repeat):
            _layers += [nn.Conv2d(out_dim,  out_dim, 
                        kernel_size = 3, padding=1, bias=True)]
            _layers += [getNormLayer(norm)(out_dim )]
            _layers += [activ]

        self.out = nn.Sequential(*_layers)    
         
    def forward(self, img_input, sent_input, hid_input):
        img_trans = self.side_trans(img_input)
        
        b, c ,_ ,_ = img_trans.size()
        row, col = img_trans.size()[2::]
        sent_input = sent_input.unsqueeze(-1).unsqueeze(-1)
        sent_input = sent_input.expand(b, c, row, col )
        #print(img_trans.size(), sent_input.size(), hid_input.size())
        com_inp = torch.cat([img_trans, sent_input, hid_input], 1)
        out = self.out(com_inp)
        return out

class Generator(nn.Module):
    def __init__(self, input_size, sent_dim,  noise_dim, num_chan, 
                 emb_dim, hid_dim, norm = 'instance',branch=True, small_output=True):
        super(Generator, self).__init__()
        self.__dict__.update(locals())
        self.register_buffer('device_id', torch.zeros(1))
        self.sent_init = condEmbedding(sent_dim, emb_dim)
        # We need to know how many layers we will use at the beginning
        side_out = 128 # map 3 channel image to this channel.
        activ = genAct()
        
        self.node_4   = sentConv(emb_dim+noise_dim, 4, 4, hid_dim, activ)
        self.node_8   = up_conv(hid_dim, hid_dim, norm, activ, repeat=1, get_layer =True)

        self.node_16   = up_conv(hid_dim, hid_dim, norm, activ, repeat=1, get_layer =True)
        #self.branch_16 = brach_out(hid_dim, 3, norm, activ, repeat=1, get_layer =True)
        #self.sent_16   = condEmbedding(sent_dim, emb_dim)
        #self.conn_16   = connectSide(3, side_out, hid_dim, emb_dim, hid_dim, norm, activ, repeat= 0)
        
        self.node_32   = up_conv(hid_dim, hid_dim, norm, activ, repeat=1, get_layer =True)
        #self.branch_32 = brach_out(hid_dim, 3, norm, activ, repeat=1, get_layer =True)
        #self.sent_32   = condEmbedding(sent_dim, emb_dim)
        #self.conn_32   = connectSide(3, side_out, hid_dim, emb_dim, hid_dim, norm, activ, repeat= 0)

        self.node_64   = up_conv(hid_dim, hid_dim, norm, activ, repeat=1, get_layer =True)
        #self.branch_64 = brach_out(hid_dim, 3, norm, activ, repeat=1, get_layer =True)
        #self.sent_64   = condEmbedding(sent_dim, emb_dim)
        #self.conn_64   = connectSide(3, side_out, hid_dim, emb_dim, hid_dim, norm, activ, repeat= 0)

        self.node_128 = up_conv(hid_dim, hid_dim, norm, activ, repeat=1, get_layer =True)
        self.node_256 = up_conv(hid_dim, hid_dim, norm, activ, repeat=1, get_layer =True)
        self.out_256  = brach_out(hid_dim, 3, norm, activ, repeat=1, get_layer =True)
        self.apply(weights_init)

    def forward(self, sent_embeddings, z=None):
        out_dict = OrderedDict()
        sent_random, kl_loss  = self.sent_init(sent_embeddings)
        com_inp = torch.cat([sent_random, z], dim=1)

        node_4 = self.node_4(com_inp)
        node_8 = self.node_8(node_4)

        node_16 = self.node_16(node_8)
        #branch_16 = self.branch_16(node_16)
        #sent_16, _this_kl  = self.sent_16(sent_embeddings)
        #kl_loss += _this_kl
        #conn_16 = self.conn_16(branch_16, sent_16, node_16)

        node_32 = self.node_32(node_16)
        #branch_32 = self.branch_32(node_32)
        #sent_32, _this_kl  = self.sent_32(sent_embeddings)
        #kl_loss += _this_kl
        #conn_32 = self.conn_32(branch_32, sent_32, node_32)

        node_64 = self.node_64(node_32)
        #branch_64 = self.branch_64(node_64)
        #sent_64, _this_kl  = self.sent_64(sent_embeddings)
        #kl_loss += _this_kl
        #conn_64 = self.conn_64(branch_64, sent_64, node_64)

        node_128 = self.node_128(node_64)
        node_256 = self.out_256(self.node_256(node_128))

        #out_dict['output_16']  = branch_16
        #out_dict['output_32']  = branch_32
        #out_dict['output_64']  = branch_64
        out_dict['output_256'] = node_256
                
        return out_dict, kl_loss

class ImageDown(torch.nn.Module):
    '''
       This module encode image to 16*16 feat maps
    '''
    def __init__(self, input_size, num_chan, hid_dim, out_dim, down_rate, norm = 'instance'):
        super(ImageDown, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.__dict__.update(locals())
        activ = discAct()
        max_down_rate = math.log(input_size, 2)
        assert down_rate <= max_down_rate, 'down rate is too large for this image'
        down_rate = min(down_rate, max_down_rate)
        _layers = []
        _layers += conv_norm(num_chan, hid_dim, norm, activ, repeat=0, get_layer = False)

        if input_size == 16:
            _layers += down_conv(hid_dim, hid_dim, norm, activ, repeat=1, get_layer = False)
            _layers += down_conv(hid_dim, hid_dim, norm, activ, repeat=1, get_layer = False)
            _layers += conv_norm(hid_dim, hid_dim, norm, activ, repeat=1, get_layer = False,
                                 kernel_size=1, padding=0)
            _layers += conv_norm(hid_dim, hid_dim, norm, activ, repeat=0, get_layer = False,
                                 kernel_size=4, padding=0)
        elif input_size == 32:
            _layers += down_conv(hid_dim, hid_dim,  norm, activ,  repeat=1, get_layer = False)  # 16
            _layers += down_conv(hid_dim, hid_dim,  norm, activ,  repeat=1, get_layer = False)  # 8
            _layers += down_conv(hid_dim, hid_dim,  norm, activ,  repeat=1, get_layer = False)  # 4
            _layers += conv_norm(hid_dim, hid_dim, norm, activ, repeat=1, get_layer = False, # 4
                                 kernel_size=1, padding=0)
            _layers += conv_norm(hid_dim, hid_dim, norm, activ, repeat=0, get_layer = False, # 1
                                 kernel_size=4, padding=0)
        else:                                                  
            for _ in range(down_rate):
                _layers += down_conv(hid_dim, hid_dim, norm, activ, repeat=1, get_layer = False)

        _layers += conv_norm(hid_dim, out_dim, norm, activ, repeat=0, get_layer = False,
                            kernel_size=1, padding=0)

        self.node = nn.Sequential(*_layers)

    def forward(self, inputs):
        # inputs (B, C, H, W), must be dividable by 32
        # return (B, C, 16, 16)
        return self.node(inputs)



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
        activ = discAct()

        _layers = [nn.Linear(sent_dim, emb_dim)]
        #_layers += [nn.BatchNorm1d(emb_dim)]
        _layers += [discAct()]
        _layers += [nn.Linear(emb_dim, emb_dim)]
        #_layers += [nn.BatchNorm1d(emb_dim*4*4)]
        _layers += [discAct()]
        self.context_emb_pipe = nn.Sequential(*_layers)

        #self.img_encoder_16  = ImageDown(16,  num_chan, hid_dim, enc_dim, 4, norm) # 1
        #self.img_encoder_32  = ImageDown(32,  num_chan, hid_dim, enc_dim, 4, norm) # 1
        #self.img_encoder_64  = ImageDown(64,  num_chan, hid_dim, enc_dim, 4, norm) # 4
        self.img_encoder_256 = ImageDown(256, num_chan, hid_dim, enc_dim, 4, norm) # 16
        
        #self.pair_disc_16  = catSentConv(enc_dim, emb_dim, 1, norm)
        #self.pair_disc_32  = catSentConv(enc_dim, emb_dim, 1, norm)
        #self.pair_disc_64  = catSentConv(enc_dim, emb_dim, 4, norm)
        self.pair_disc_256 = catSentConv(enc_dim, emb_dim, 16, norm)
        
        #self.img_disc_16  = conv_norm(enc_dim, 1, norm, activ, s, True, True, 1, 0)
        #self.img_disc_32  = conv_norm(enc_dim, 1, norm, activ, s, True, True, 1, 0)
        #self.img_disc_64  = conv_norm(enc_dim, 1, norm, activ, s, True, True, 1, 0)
        self.img_disc_256 = conv_norm(enc_dim, 1, norm, activ, 2, True, True, 1, 0)
    
    def forward(self, images, embdding):
        '''
        images: (B, C, H, W)
        embdding : (B, sent_dim)
        outptuts:
        -----------
        img_code B*chan*col*row
        pair_disc_out: B*1
        img_disc: B*1*col*row
        '''
        out_dict = OrderedDict()
        img_size = images.size()[3]
        assert img_size in [16, 32, 64, 256], 'wrong input size {} in image discriminator'.format(img_size)
        
        img_encoder_sym  = 'img_encoder_{}'.format(img_size)
        img_disc_sym = 'img_disc_{}'.format(img_size)
        pair_disc_sym = 'pair_disc_{}'.format(img_size)

        img_encoder = getattr(self, img_encoder_sym)
        img_disc    = getattr(self, img_disc_sym)
        pair_disc   = getattr(self, pair_disc_sym)

        sent_code = self.context_emb_pipe(embdding)
        
        img_code  = img_encoder(images)
        
        pair_disc_out = pair_disc(sent_code, img_code)
        img_disc_out  = img_disc(img_code)
        
        out_dict['pair_disc']     = pair_disc_out
        out_dict['img_disc']      = img_disc_out
        out_dict['content_code']  = img_code
        return out_dict

