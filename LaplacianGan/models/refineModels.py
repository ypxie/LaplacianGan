import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict
from ..proj_utils.model_utils import *
import math

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

def cat_vec_conv(text_enc, img_enc):
    # text_enc (B, dim)
    # img_enc  (B, chn, row, col)
    b, c = text_enc.size()
    row, col = img_enc.size()[2::]
    text_enc = text_enc.unsqueeze(-1).unsqueeze(-1)
    text_enc = text_enc.expand(b, c, row, col )
    com_inp = torch.cat([img_enc, text_enc], dim=1)
    return com_inp

class Generator(nn.Module):
    def __init__(self, input_size, sent_dim,  noise_dim, num_chan, emb_dim, hid_dim, 
                 norm = 'ln', side_list=[64, 128]):
        super(Generator, self).__init__()
        self.__dict__.update(locals())
        self.register_buffer('device_id', torch.zeros(1))
        self.condEmbedding = condEmbedding(sent_dim, emb_dim)
        # We need to know how many layers we will use at the beginning
        
        activ = genAct()
        fake_active = passthrough()

        node4_0 = sentConv(emb_dim+noise_dim, 4, 4, self.hid_dim*8, None, False)
        _layers = conv_norm(self.hid_dim*8,  self.hid_dim*2, norm,  activ, 0, False,True,   1, 0)
        #_layers += conv_norm(self.hid_dim*2,  self.hid_dim*2, norm, activ, 0, False,True,   1, 0)
        _layers += conv_norm(self.hid_dim*2,  self.hid_dim*8, norm, activ, 0, False,False,  3, 1)
        node4_1 = nn.Sequential(*_layers)
        self.node_4  = resConn(node4_0, node4_1, activ)

        _layers = [nn.Upsample((8,8),mode='nearest')]
        _layers += conv_norm(self.hid_dim*8,  self.hid_dim*4, norm, activ, 0, False,False,  3, 1)
        node8_0 = nn.Sequential(*_layers)
        _layers = conv_norm(self.hid_dim*4,  self.hid_dim*1, norm,  activ, 0, False,True,  1, 0)
        #_layers += conv_norm(self.hid_dim*1,  self.hid_dim*1, norm,  activ, 0, False,False, 1, 0)
        _layers += conv_norm(self.hid_dim*1,  self.hid_dim*4, norm,  activ, 0, False,False, 3, 1)
        node8_1 = nn.Sequential(*_layers)
        self.node_8  = resConn(node8_0, node8_1, activ)
        
        _layers = [nn.Upsample((16,16),mode='nearest')]
        _layers += conv_norm(self.hid_dim*4,  self.hid_dim*2, norm, activ, 0, False,False,  3, 1)
        node4_0 = nn.Sequential(*_layers)
        _layers = conv_norm(self.hid_dim*2,  self.hid_dim*1, norm,   activ, 0, False,True,  1, 0)
        #_layers += conv_norm(self.hid_dim*1,  self.hid_dim*1, norm,  activ, 0, False,False, 1, 0)
        _layers += conv_norm(self.hid_dim*1,  self.hid_dim*2, norm,  activ, 0, False,False, 3, 1)
        node4_1 = nn.Sequential(*_layers)
        self.node_16  = resConn(node4_0, node4_1, activ)
        
        _layers = [nn.Upsample((32,32),mode='nearest')]
        _layers += conv_norm(self.hid_dim*2,  self.hid_dim*2, norm, activ, 0, False,False,  3, 1)
        node32_0 = nn.Sequential(*_layers)
        _layers = conv_norm(self.hid_dim*2,  self.hid_dim*1, norm,   activ, 0, False,True,  1, 0)
        #_layers += conv_norm(self.hid_dim*1,  self.hid_dim*1, norm,  activ, 0, False,False, 1, 0)
        _layers += conv_norm(self.hid_dim*1,  self.hid_dim*2, norm,  activ, 0, False,False, 3, 1)
        node32_1 = nn.Sequential(*_layers)
        self.node_32  = resConn(node32_0, node32_1, activ)

        _layers = [nn.Upsample((64,64),mode='nearest')]
        _layers += conv_norm( hid_dim*2,  hid_dim, norm, activ, 1, False, True,  3, 1)
        self.node_64 = nn.Sequential(*_layers)
        #node64_0 = nn.Sequential(*_layers)
        #_layers = conv_norm( hid_dim*1, hid_dim//2, norm,   activ, 0, False,True,  1, 0)
        #_layers += conv_norm( hid_dim//2,   hid_dim, norm,  activ, 0, False,False, 3, 1)
        #node64_1 = nn.Sequential(*_layers)
        #self.node_64  = resConn(node64_0, node64_1, activ)
        if 64 in side_list:
            #self.condEmbedding_64 = condEmbedding(sent_dim, emb_dim)
            self.out_64  = brach_out(hid_dim, 3, norm, activ, repeat = 1, get_layer =True)
            self.side_64 = connectSideBefore(3, hid_dim//2, hid_dim, hid_dim*2, hid_dim, norm, activ, 2)

        _layers = [nn.Upsample((128, 128), mode='nearest')]
        _layers += conv_norm(hid_dim,  hid_dim, norm, activ, 1, False, True,  3, 1)
        self.node_128 = nn.Sequential(*_layers)
        
        #node64_0 = nn.Sequential(*_layers)
        #_layers = conv_norm(hid_dim//2,  hid_dim//4, norm,   activ, 0, False,True,  1, 0)
        #_layers += conv_norm(hid_dim//4,  hid_dim//2, norm,  activ, 0, False,False, 3, 1)
        #node64_1 = nn.Sequential(*_layers)
        #self.node_128  = resConn(node64_0, node64_1, activ)
        if 128 in side_list:
            #self.condEmbedding_128 = condEmbedding(sent_dim, emb_dim)
            self.out_128  = brach_out(hid_dim, 3, norm, activ, repeat=1, get_layer =True)
            self.side_128 = connectSideBefore(3, hid_dim//2, hid_dim, hid_dim*2, hid_dim, norm, activ, 2)
            
        _layers = [nn.Upsample((256, 256), mode='nearest')]
        _layers += conv_norm(hid_dim,  hid_dim//2, norm, activ, 1, False, True,  3, 1)
        self.node_256 = nn.Sequential(*_layers)
        #_layers += conv_norm(hid_dim//2,  hid_dim//4, norm, activ, 0, False,False,  3, 1)
        #node256_0 = nn.Sequential(*_layers)
        #_layers = conv_norm(hid_dim//4,  hid_dim//4, norm,   activ, 0, False,True,  1, 0)
        #_layers += conv_norm(hid_dim//4,  hid_dim//4, norm,  activ, 0, False,False, 3, 1)
        #node256_1 = nn.Sequential(*_layers)
        #self.node_256  = resConn(node256_0, node256_1, activ)
        
        self.out_256  = brach_out(hid_dim//2, 3, norm, activ, repeat=1, get_layer =True)
        
        self.apply(weights_init)

    def forward(self, sent_embeddings, z=None):
        out_dict = OrderedDict()
        sent_random, kl_loss  = self.condEmbedding(sent_embeddings)

        com_inp = torch.cat([sent_random, z], dim=1)
        node_4 = self.node_4(com_inp)
        node_8 = self.node_8(node_4)
        
        node_16 = self.node_16(node_8)
        node_32 = self.node_32(node_16)

        node_64 = self.node_64(node_32)

        if 64 in self.side_list:
            #sent_random, _kl_loss  = self.condEmbedding_64(sent_embeddings)
            out_64  = self.out_64(node_64)
            node_64 = self.side_64(Variable(out_64.data), node_16, node_64)
            out_dict['output_64']  = out_64
            #kl_loss += _kl_loss

        node_128 = self.node_128(node_64)

        if 128 in self.side_list:
            #sent_random, _kl_loss  = self.condEmbedding_128(sent_embeddings)
            out_128  = self.out_128(node_128)
            node_128 = self.side_128(Variable(out_128.data), node_32, node_128)
            out_dict['output_128']  = out_128
            #kl_loss += _kl_loss

        node_256 = self.node_256(node_128)

        out_256 = self.out_256(node_256)

        out_dict['output_256']  = out_256

        return out_dict, kl_loss

class ImageDown(torch.nn.Module):
    '''
       This module encode image to 16*16 feat maps
    '''
    def __init__(self, input_size, num_chan, hid_dim, out_dim, down_rate, norm = 'instance'):
        super(ImageDown, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        activ = discAct()
        self.__dict__.update(locals())
        max_down_rate = math.log(input_size, 2)
        assert down_rate <= max_down_rate, 'down rate is too large for this image'

        if input_size == 64:
            _layers = conv_norm(num_chan, hid_dim,     norm,  activ, 0, False,True, 3, 1, 2) # 32
            _layers += conv_norm(hid_dim, hid_dim*2,   norm,  activ, 0, False,True, 3, 1, 2) # 16
            _layers += conv_norm(hid_dim*2, out_dim,   norm,  activ, 0, False,True, 3, 1, 2) # 8
            _layers += conv_norm(out_dim, out_dim,   norm,    activ, 0, False,True, 5, 0, 1) # 4
            #self.node_0  = nn.Sequential(*_layers)
            #self.node_1 = conv_norm(out_dim, out_dim,   norm,  activ,  2, True,False, 1,0,1)

        if input_size == 128:
            _layers  = conv_norm(num_chan, hid_dim*2,  norm, activ, 0, False,True, 3,1,2)  # 64
            _layers += conv_norm(hid_dim*2, hid_dim*2,  norm, activ, 0, False,True,  3,1,2)  # 32
            _layers += conv_norm(hid_dim*2, hid_dim*4,  norm, activ, 0, False,True,  3,1,2)  # 16
            _layers += conv_norm(hid_dim*4, out_dim,  norm, activ, 0, False,True,  3,1,2)  # 8  
            #self.node_0  = nn.Sequential(*_layers)
            #self.node_1 = conv_norm(out_dim, out_dim,   norm,  activ,  2, True,False, 1,0,1)
        
        if input_size == 256:
            _layers  = conv_norm(num_chan, hid_dim*2,  norm, activ, 0, False,True, 3,1,2)  # 128
            _layers += conv_norm(hid_dim*2, hid_dim*2,  norm, activ, 0, False,True,  3,1,2)  # 64
            _layers += conv_norm(hid_dim*2, hid_dim*4,  norm, activ, 0, False,True,  3,1,2)  # 32
            _layers += conv_norm(hid_dim*4, out_dim,  norm, activ, 0, False,True,  3,1,2)  # 16 
            #_layers += conv_norm(out_dim, out_dim,  norm, activ, 0, False,True,  3,1,2)  # 8   
            #self.node_0  = nn.Sequential(*_layers)
            #self.node_1 = conv_norm(out_dim, out_dim,   norm,  activ,  2, True,False, 1,0,1)
        self.node = nn.Sequential(*_layers)

    def forward(self, inputs):
        # inputs (B, C, H, W), must be dividable by 32
        # return (B, C, row, col), and content_code
        content_code = self.node(inputs)
        #node_1 = self.node_1(content_code)
        #output =  self.activ(content_code + node_1)
        return content_code, content_code

class Discriminator(torch.nn.Module):
    '''
    enc_dim: Reduce images inputs to (B, enc_dim, H, W)
    emb_dim: The sentence embedding dimension.
    '''

    def __init__(self, input_size, num_chan,  hid_dim, 
                sent_dim, emb_dim, norm='ln',side_list=[64, 128]):
        super(Discriminator, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.__dict__.update(locals())
        activ = discAct()

        _layers = [nn.Linear(sent_dim, emb_dim)]
        _layers += [discAct()]
        self.context_emb_pipe = nn.Sequential(*_layers)

        if 64 in side_list:
            enc_dim = hid_dim
            self.img_encoder_64   = ImageDown(64,  num_chan,  hid_dim,  enc_dim, 4, norm)  # 8
            self.pair_disc_64   = catSentConv(enc_dim, emb_dim, 4,  norm, activ, 0)

            _layers  = conv_norm(enc_dim, hid_dim*4,   norm,  activ, 1, False, True, 1,0,1)
            _layers += conv_norm(hid_dim*4, hid_dim*2,   norm,  activ, 0, False,True, 1,0,1)
            _layers += [nn.Conv2d(hid_dim*2, 1, kernel_size = 4, padding = 0, bias=True)]   # 1
            self.img_disc_64 = nn.Sequential(*_layers)
            
        if 128 in side_list:
            enc_dim = hid_dim*2
            self.img_encoder_128  = ImageDown(128,  num_chan, hid_dim,  enc_dim, 4, norm)  # 8
            self.pair_disc_128  = catSentConv(enc_dim, emb_dim, 8,  norm, activ, 0)
            _layers  = conv_norm(enc_dim, hid_dim*4,   norm,  activ, 1, False, True,  1,0,1)
            _layers += conv_norm(hid_dim*4, hid_dim*2,   norm,  activ, 0, False,True, 1,0,1)
            _layers += [nn.Conv2d(hid_dim*2, 1, kernel_size = 2, padding = 0, bias=True)]  # 7
            self.img_disc_128 = nn.Sequential(*_layers)
        
        enc_dim = hid_dim*2
        self.img_encoder_256  = ImageDown(256, num_chan,  hid_dim,  enc_dim, 4, norm)  # 16
        self.pair_disc_256  = catSentConv(enc_dim, emb_dim, 16,  norm, activ, 1)
        
        _layers  = conv_norm(enc_dim, hid_dim*4,   norm,  activ, 1, False, True, 1,0,1)
        _layers += conv_norm(hid_dim*4, hid_dim*2,   norm,  activ, 0, False,True, 1,0,1)
        _layers += [padConv2d(hid_dim*2, 1, kernel_size = 1, padding=0, bias=True)]   # 16
        self.img_disc_256 = nn.Sequential(*_layers)
        self.apply(weights_init)
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
        assert img_size in [32, 64, 128, 256], 'wrong input size {} in image discriminator'.format(img_size)
        
        img_encoder_sym  = 'img_encoder_{}'.format(img_size)
        img_disc_sym = 'img_disc_{}'.format(img_size)
        pair_disc_sym = 'pair_disc_{}'.format(img_size)

        img_encoder = getattr(self, img_encoder_sym)
        img_disc    = getattr(self, img_disc_sym)
        pair_disc   = getattr(self, pair_disc_sym)

        sent_code = self.context_emb_pipe(embdding)
        
        img_code, content_code = img_encoder(images)
        
        pair_disc_out = pair_disc(sent_code, img_code)
        img_disc_out  = img_disc(img_code)
        
        out_dict['pair_disc']     = pair_disc_out
        out_dict['img_disc']      = img_disc_out
        out_dict['content_code']  = content_code
        return out_dict
