import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict
from ..proj_utils.model_utils import to_device, getNormLayer, weights_init
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
    epsilon = to_device( torch.randn(mean.size()), mean, requires_grad=False) 
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
        out = F.leaky_relu( self.linear(inputs), 0.2, inplace=True )
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]

        c, kl_loss = sample_encoded_context(mean, log_sigma, kl_loss)
        return c, kl_loss

def genAct():
    return nn.ReLU(True)
def discAct():
    return nn.LeakyReLU(0.2, True)
def get_activation_layer(name):
    if name == 'lrelu':
        act_layer = nn.LeakyReLU(0.2, inplace=True)
    else:
        act_layer = nn.ReLU(True)
    return act_layer
    

def pad_conv_norm(dim_in, dim_out, norm_layer, kernel_size=3, use_activation=True, use_bias=False, activation=nn.ReLU(True)):
    # designed for generators
    seq = []
    if kernel_size != 1:
        seq += [nn.ReflectionPad2d(1)]

    seq += [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=0, bias=use_bias),
           norm_layer(dim_out)]
    
    if use_activation:
        seq += [activation]
    
    return nn.Sequential(*seq)

def conv_norm(dim_in, dim_out, norm_layer, kernel_size=3, stride=1, use_activation=True, use_bias=False, activation=nn.ReLU(True), use_norm=True):
    # designed for discriminator
    if kernel_size == 3:
        padding = 1
    else:
        padding = 0

    seq = [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=use_bias, stride=stride),
           ]
    if use_norm:
        seq += [norm_layer(dim_out)]
    if use_activation:
        seq += [activation]
    
    return nn.Sequential(*seq)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', use_bias=False):
        super(ResnetBlock, self).__init__()
        norm_layer = getNormLayer(norm)
        activ = get_activation_layer(activation)
        seq = [pad_conv_norm(dim, dim, norm_layer, use_bias=use_bias, activation=activ), 
               pad_conv_norm(dim, dim, norm_layer, use_activation=False, use_bias=use_bias)]
        self.res_block = nn.Sequential(*seq)

    def forward(self, input):
        # TODO do we need to add activation? 
        # CycleGan regards this. I guess to prevent spase gradients
        
        return self.res_block(input) + input


class MultiModalBlock(nn.Module):
    def __init__(self, text_dim, img_dim, norm, activation='relu', use_bias=False, upsample_factor=3):
        super(MultiModalBlock, self).__init__()
        norm_layer = getNormLayer(norm)
        activ = get_activation_layer(activation)
        # upsampling 2^3 times
        seq = []
        cur_dim = text_dim
        for i in range(upsample_factor):
            seq += [nn.Upsample(scale_factor=2, mode='nearest')]
            seq += [pad_conv_norm(cur_dim, cur_dim//2, norm_layer, activation=activ)]
            cur_dim = cur_dim//2

        self.upsample_path = nn.Sequential(*seq)
        self.joint_path = nn.Sequential(*[
            pad_conv_norm(cur_dim+img_dim, img_dim, norm_layer, kernel_size=1, use_activation=False)
        ])
    def forward(self, text, img ):
        upsampled_text = self.upsample_path(text)
        
        out = self.joint_path(torch.cat([img, upsampled_text],1))
        return out


class MultiModalBlock2(nn.Module):
    def __init__(self, text_dim, img_dim, norm, activation='relu', use_bias=False, upsample_factor=3):
        super(MultiModalBlock2, self).__init__()
        norm_layer = getNormLayer(norm)
        activ = get_activation_layer(activation)
        # upsampling 2^3 times
        seq = []
        cur_dim = text_dim
        for i in range(upsample_factor):
            seq += [nn.Upsample(scale_factor=2, mode='nearest')]
            seq += [pad_conv_norm(cur_dim, cur_dim//2, norm_layer, activation=activ)]
            cur_dim = cur_dim//2

        self.upsample_path = nn.Sequential(*seq)
        self.joint_path = nn.Sequential(*[
            pad_conv_norm(cur_dim+img_dim, img_dim, norm_layer, kernel_size=1, use_activation=False)
        ])
    def forward(self, text, img ):
        # text is  [B, 128]
        # img is 
        upsampled_text = self.upsample_path(text)
        
        out = self.joint_path(torch.cat([img, upsampled_text],1))
        return out

class sentConv2(nn.Module):
    def __init__(self, in_dim, row, col, channel, norm='bn',
                 activ = None, last_active = False):
        super(sentConv2, self).__init__()
        self.__dict__.update(locals())
        out_dim = row*col*channel
        norm_layer = getNormLayer(norm, dim=1)
        _layers = [nn.Linear(in_dim, out_dim)]
        _layers += [norm_layer(out_dim)]

        if activ is not None:
            _layers += [activ] 
        
        self.out = nn.Sequential(*_layers)    
         
    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output



class Generator(nn.Module):
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, norm='bn', activation='relu',
                 output_size=256):
        super(Generator, self).__init__()
        self.__dict__.update(locals())
        norm_layer = getNormLayer(norm)
        act_layer = get_activation_layer(activation)


        self.register_buffer('device_id', torch.IntTensor(1))
        self.condEmbedding = condEmbedding(sent_dim, emb_dim)
        self.vec_to_tensor = sentConv2(emb_dim+noise_dim, 4, 4, self.hid_dim*8, norm=norm)
        
        '''user defefined'''
        self.output_size = output_size
        # 64, 128, or 256 version
        
        if output_size == 256:
            num_scales = [4, 8, 16, 32, 64, 128, 256]
            text_upsampling_at = [4, 8, 16] 
        elif output_size == 64:
            num_scales = [4, 8, 16, 32, 64]
            text_upsampling_at = [4] 
        elif output_size == 128:
            num_scales = [4, 8, 16, 32, 64, 128]
            text_upsampling_at = [4, 8] 

        print ('>> initialized a {} size generator'.format(output_size))

        reduce_dim_at = [8, 64, 256] 
        side_output_at = [64, 128, 256] 
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
                seq += [pad_conv_norm(cur_dim, cur_dim//2, norm_layer, activation=act_layer)]
                cur_dim = cur_dim//2
            # print ('scale {} cur_dim {}'.format(num_scales[i], cur_dim))
            # add residual blocks
            for n in range(num_resblock):
                seq += [ResnetBlock(cur_dim, norm, activation=activation)]
            # add main convolutional module
            setattr(self, 'scale_%d'%(num_scales[i]), nn.Sequential(*seq) )

            # add upsample module to concat with upper layers 
            if num_scales[i] in text_upsampling_at:
                ## img_dim // 2 is bacause 
                setattr(self, 'upsample_%d'%(num_scales[i]), MultiModalBlock(text_dim=cur_dim, img_dim=cur_dim//2, norm=norm, activation=activation))
            # configure side output module
            if num_scales[i] in side_output_at:
                setattr(self, 'tensor_to_img_%d'%(num_scales[i]), branch_out2(cur_dim))

        
        self.apply(weights_init)

    def forward(self, sent_embeddings, z):
        # sent_embeddings: [B, 1024]
        out_dict = OrderedDict()
        sent_random, kl_loss  = self.condEmbedding(sent_embeddings) # sent_random [B, 128]
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


class GeneratorNoSkip(Generator):
    # very simple skip connection to embed text code in multiscale outputs
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, norm='bn', activation='relu',
                 output_size=256):
        super(GeneratorNoSkip, self).__init__(sent_dim, noise_dim, emb_dim, hid_dim, norm, activation, output_size)
        print ('WARNING: GeneratorNoSkip version without upsample_32')
        delattr(self, 'upsample_4')
        if hasattr(self, 'upsample_8'):
            delattr(self, 'upsample_8')
        if hasattr(self, 'upsample_16'):
            delattr(self, 'upsample_16')
            

    def forward(self, sent_embeddings, z):
        # sent_embeddings: [B, 1024]
        out_dict = OrderedDict()
        sent_random, kl_loss  = self.condEmbedding(sent_embeddings) # sent_random [B, 128]
        text = torch.cat([sent_random, z], dim=1)

        x = self.vec_to_tensor(text)
        x_4 = self.scale_4(x)
        x_8 = self.scale_8(x_4)
        x_16 = self.scale_16(x_8)
        x_32 = self.scale_32(x_16)
        
        # skip 4x4 feature map to 32 and send to 64
        x_64 = self.scale_64(x_32)
        out_dict['output_64'] = self.tensor_to_img_64(x_64)
        
        # if self.output_size > 64:
        #     # skip 8x8 feature map to 64 and send to 128
        #     x_64_8 = self.upsample_8(x_8, x_64)
        #     x_128 = self.scale_128(x_64_8)
        #     out_dict['output_128'] = self.tensor_to_img_128(x_128)

        # if self.output_size > 128:
        #     # skip 16x16 feature map to 128 and send to 256
        #     x_128_16 = self.upsample_16(x_16, x_128)
        #     out_256 = self.scale_256(x_128_16)

        #     out_dict['output_256'] = self.tensor_to_img_256(out_256)

        return out_dict, kl_loss

class GeneratorSimpleSkip(Generator):
    # very simple skip connection to embed text code in multiscale outputs
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, norm='bn', activation='relu',
                 output_size=256):
        super(GeneratorSimpleSkip, self).__init__(sent_dim, noise_dim, emb_dim, hid_dim, norm, activation, output_size)
        self.upsample_4 = None
        if self.output_size > 64:
            self.upsample_8  = None
        if self.output_size > 128:
            self.upsample_16 = None

    def forward(self, sent_embeddings, z=None):
        # sent_embeddings: [B, 1024]
        
        out_dict = OrderedDict()
        sent_random, kl_loss  = self.condEmbedding(sent_embeddings)
        text = torch.cat([sent_random, z], dim=1)
        
        # replicate sent_random
        text_hidden =  sent_random.unsqueeze(-1).unsqueeze(-1)
        b, text_dim = sent_random.size()

        x = self.vec_to_tensor(text)
        x_4 = self.scale_4(x)
        x_8 = self.scale_8(x_4)
        x_16 = self.scale_16(x_8)
        x_32 = self.scale_32(x_16)
        
        # concat text_hiddent to x_32
        x_32_4 = torch.cat([x_32, text_hidden.expand((b,text_dim,32,32))], 1)
        x_64 = self.scale_64(x_32_4)
        out_dict['output_64'] = self.tensor_to_img_64(x_64)
        
        if self.output_size > 64:
            # skip 8x8 feature map to 64 and send to 128
            x_64_8 = torch.cat([x_64, text_hidden.expand((b,text_dim,64,64))], 1)
            x_128 = self.scale_128(x_64_8)
            out_dict['output_128'] = self.tensor_to_img_128(x_128)

        if self.output_size > 128:
            # skip 16x16 feature map to 128 and send to 256
            x_128_16 = torch.cat([x_128, text_hidden.expand((b,text_dim,128,128))], 1)
            out_256 = self.scale_256(x_128_16)

            out_dict['output_256'] = self.tensor_to_img_256(out_256)

        return out_dict, kl_loss  


class ImageDown(torch.nn.Module):
    '''
       This module encode image to 16*16 feat maps
    '''
    def __init__(self, input_size, num_chan, out_dim, norm='norm'):
        super(ImageDown, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        
        self.__dict__.update(locals())
        norm_layer = getNormLayer(norm)
        activ = discAct()
        _layers = []
        if input_size == 64:
            cur_dim = 128 # for testing
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 32
            _layers += [conv_norm(cur_dim, cur_dim*2,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*2, cur_dim*4,  norm_layer, stride=2, activation=activ)] # 8
            _layers += [conv_norm(cur_dim*4, out_dim,  norm_layer, stride=2, activation=activ)] # 4
            
            # add more layers like StackGAN did. I don't think it is necessary.
            # cur_dim = cur_dim * 4
            # _layers = []
            # _layers += [conv_norm(cur_dim, cur_dim//4,  norm_layer, kernel_size=1, activation=activ)] # 16
            # _layers += [conv_norm(cur_dim//4, cur_dim//4, norm_layer, activation=activ)] # 16
            # _layers += [conv_norm(cur_dim//4, out_dim, norm_layer,  use_activation=False)] # 16
            # self.skip = nn.Sequential(*_layers)
        if input_size == 128:
            cur_dim = 64 # for testing
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 32
            _layers += [conv_norm(cur_dim, cur_dim*2,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*2, cur_dim*4,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*4, cur_dim*8,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*8, out_dim,  norm_layer, stride=2, activation=activ)] # 16
        
        if input_size == 256:
            cur_dim = 32 # for testing
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 32
            _layers += [conv_norm(cur_dim, cur_dim*2,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*2, cur_dim*4,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*4, cur_dim*8,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*8, cur_dim*16,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*16, out_dim,  norm_layer, stride=2, activation=activ)] # 16
            

        self.node = nn.Sequential(*_layers)

    def forward(self, inputs):
        # inputs (B, C, H, W), must be dividable by 32
        # return (B, C, row, col), and content_code
        out = self.node(inputs)
        # x = self.node(inputs)
        # out = F.leaky_relu(x + self.skip(x), 0.2, inplace=True)
        #node_1 = self.node_1(content_code)
        #output =  self.activ(content_code + node_1)
        return out

class shareImageDown(torch.nn.Module):
    '''
       This module encode image to 16*16 feat maps
    '''
    def __init__(self, input_size, num_chan, hid_dim, out_dim, norm='norm', shared_block=None):
        super(shareImageDown, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        
        self.__dict__.update(locals())
        norm_layer = getNormLayer(norm)
        activ = discAct()
        cur_dim = 128

        # Shared 32*32 feature encoder
        
        _layers = []

        if input_size == 64:      
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 32
        if input_size == 128:
            
            _layers += [conv_norm(num_chan, cur_dim//2, norm_layer, stride=2, activation=activ, use_norm=False)] # 64
            _layers += [conv_norm(cur_dim//2, cur_dim,  norm_layer, stride=2, activation=activ)] # 32
        
        if input_size == 256:
            
            _layers += [conv_norm(num_chan, cur_dim//4, norm_layer, stride=2, activation=activ, use_norm=False)] # 128
            _layers += [conv_norm(cur_dim//4, cur_dim//2,  norm_layer, stride=2, activation=activ)] # 64
            _layers += [conv_norm(cur_dim//2, cur_dim,  norm_layer, stride=2, activation=activ)] # 32
            
        self.node = nn.Sequential(*_layers)
        self.shared_block = shared_block

    def forward(self, inputs):
        # inputs (B, C, H, W), must be dividable by 32
        # return (B, C, row, col), and content_code
        out = self.node(inputs)
        out = self.shared_block(out)
        return out
        
class shareImageDown(torch.nn.Module):
    '''
       This module encode image to 16*16 feat maps
    '''
    def __init__(self, input_size, num_chan, hid_dim, out_dim, norm='norm', shared_block=None):
        super(shareImageDown, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        
        self.__dict__.update(locals())
        norm_layer = getNormLayer(norm)
        activ = discAct()
        cur_dim = 128

        # Shared 32*32 feature encoder
        
        _layers = []

        if input_size == 64:      
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 32

        if input_size == 128:
            
            _layers += [conv_norm(num_chan, cur_dim//2, norm_layer, stride=2, activation=activ, use_norm=False)] # 64
            _layers += [conv_norm(cur_dim//2, cur_dim,  norm_layer, stride=2, activation=activ)] # 32
            
        
        if input_size == 256:
            
            _layers += [conv_norm(num_chan, cur_dim//4, norm_layer, stride=2, activation=activ, use_norm=False)] # 128
            _layers += [conv_norm(cur_dim//4, cur_dim//2,  norm_layer, stride=2, activation=activ)] # 64
            _layers += [conv_norm(cur_dim//2, cur_dim,  norm_layer, stride=2, activation=activ)] # 32
            
        self.node = nn.Sequential(*_layers)
        self.shared_block = shared_block

    def forward(self, inputs):
        # inputs (B, C, H, W), must be dividable by 32
        # return (B, C, row, col), and content_code
        out = self.node(inputs)
        out = self.shared_block(out)
        # x = self.node(inputs)
        # out = F.leaky_relu(x + self.skip(x), 0.2, inplace=True)
        #node_1 = self.node_1(content_code)
        #output =  self.activ(content_code + node_1)
        return out

class DiscClassifier(nn.Module):
    def __init__(self, enc_dim, emb_dim, feat_size, norm, activ):
        '''
          enc_dim: B*enc_dim*H*W
          emb_dim: the dimension of feeded embedding
          feat_size: the feature map size of the feature map. 
        '''
        super(DiscClassifier, self).__init__()
        self.__dict__.update(locals())
        norm_layer = getNormLayer(norm)
        activ = discAct()
        inp_dim = enc_dim + emb_dim
        new_feat_size = feat_size

        # TODO: do we need anyother convolutional layer to joint image-text feature.
        # Now I added. It is different from previous verison.
        _layers =  [ conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1, activation=activ),
                     nn.Conv2d(enc_dim, 1, kernel_size=new_feat_size, padding=0, bias=True)]
        ## _layers = [nn.Conv2d(inp_dim, 1, kernel_size=new_feat_size, padding=0, bias=True)]
        self.node = nn.Sequential(*_layers)

    def forward(self,sent_code,  img_code):
        sent_code =  sent_code.unsqueeze(-1).unsqueeze(-1)
        dst_shape = list(sent_code.size())
        #print(dst_shape, img_code.size())
        dst_shape[1] =  sent_code.size()[1]
        dst_shape[2] =  img_code.size()[2] 
        dst_shape[3] =  img_code.size()[3] 
        sent_code = sent_code.expand(dst_shape)
        #sent_code = sent_code.view(*dst_shape)
        #print(img_code.size(), sent_code.size())
        comp_inp = torch.cat([img_code, sent_code], dim=1)
        output = self.node(comp_inp)
        chn  = output.size()[1]
        output = output.view(-1, chn)

        return output

class Discriminator(torch.nn.Module):
    '''
    enc_dim: Reduce images inputs to (B, enc_dim, H, W)
    emb_dim: The sentence embedding dimension.
    '''

    def __init__(self, input_size, num_chan,  hid_dim, sent_dim, emb_dim, norm='bn'):
        super(Discriminator, self).__init__()
        self.register_buffer('device_id', torch.IntTensor(1))
        self.__dict__.update(locals())
        activ = discAct()
        norm_layer = getNormLayer(norm)

        _layers = [nn.Linear(sent_dim, emb_dim)]
        _layers += [activ]
        self.context_emb_pipe = nn.Sequential(*_layers)

        enc_dim = hid_dim * 4 # the ImageDown output dimension

        _layers = []
        self.img_encoder_64   = ImageDown(64,  num_chan,  enc_dim, norm)  # 4x4
        self.pair_disc_64   = DiscClassifier(enc_dim, emb_dim, feat_size=4, norm=norm, activ=activ)
        _layers =  [nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)]
        self.img_disc_64 = nn.Sequential(*_layers)
        self.max_out_size = 64

        if input_size > 64:
            self.img_encoder_128  = ImageDown(128,  num_chan, enc_dim, norm)  # 8
            self.pair_disc_128  = DiscClassifier(enc_dim, emb_dim, feat_size=4,  norm=norm, activ=activ)
            _layers = [nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)]   # 4
            self.img_disc_128 = nn.Sequential(*_layers)
            self.max_out_size = 128
            
        if input_size > 128:
            self.img_encoder_256  = ImageDown(256, num_chan, enc_dim, norm)  # 8
            self.pair_disc_256  = DiscClassifier(enc_dim, emb_dim, feat_size=4,  norm=norm, activ=activ)
            _layers = [nn.Conv2d(enc_dim, 1, kernel_size=4, padding = 0, bias=True)]   # 4
            self.img_disc_256 = nn.Sequential(*_layers)
            self.max_out_size = 256
        
        self.apply(weights_init)
        print ('>> initialized a {} size discriminator'.format(input_size))

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
        assert self.max_out_size >= img_size, 'image size {} exceeds expected maximum size {}'.format(img_size, self.max_out_size)

        img_encoder_sym  = 'img_encoder_{}'.format(img_size)
        img_disc_sym = 'img_disc_{}'.format(img_size)
        pair_disc_sym = 'pair_disc_{}'.format(img_size)

        img_encoder = getattr(self, img_encoder_sym)
        img_disc    = getattr(self, img_disc_sym)
        pair_disc   = getattr(self, pair_disc_sym)

        sent_code = self.context_emb_pipe(embdding)
        
        img_code = img_encoder(images)
        pair_disc_out = pair_disc(sent_code, img_code)
        img_disc_out  = img_disc(img_code)
        
        out_dict['pair_disc']     = pair_disc_out
        out_dict['img_disc']      = img_disc_out
        out_dict['content_code']  = img_code # useless
        return out_dict



class sharedDiscriminator(torch.nn.Module):
    '''
    enc_dim: Reduce images inputs to (B, enc_dim, H, W)
    emb_dim: The sentence embedding dimension.
    '''

    def __init__(self, input_size, num_chan,  hid_dim, sent_dim, emb_dim, norm='bn'):
        super(sharedDiscriminator, self).__init__()
        self.register_buffer('device_id', torch.IntTensor(1))
        self.__dict__.update(locals())
        activ = discAct()
        norm_layer = getNormLayer(norm)

        _layers = [nn.Linear(sent_dim, emb_dim)]
        _layers += [activ]
        self.context_emb_pipe = nn.Sequential(*_layers)

        enc_dim = hid_dim * 4 # the ImageDown output dimension

        _layers = []
        cur_dim, out_dim = 128, enc_dim
        norm_layer = getNormLayer(norm)
        _layers += [conv_norm(cur_dim, cur_dim*2,  norm_layer, stride=2, activation=activ)] # 32->16
        _layers += [conv_norm(cur_dim*2, cur_dim*4,  norm_layer, stride=2, activation=activ)] # 8
        _layers += [conv_norm(cur_dim*4, out_dim,  norm_layer, stride=2, activation=activ)] # 4
        shared_block  = nn.Sequential(*_layers)

        _layers = []
        self.img_encoder_64   = sharedImageDown(64,  num_chan,  enc_dim, norm,shared_block)  # 4x4
        self.pair_disc_64   = DiscClassifier(enc_dim, emb_dim, feat_size=4, norm=norm, activ=activ)
        _layers = [conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1, activation=activ),
                   nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)]
        self.img_disc_64 = nn.Sequential(*_layers)
        self.max_out_size = 64

        if input_size > 64:
            self.img_encoder_128  = shareImageDown(128,  num_chan, enc_dim, norm, shared_block)  # 8
            self.pair_disc_128  = DiscClassifier(enc_dim, emb_dim, feat_size=4,  norm=norm, activ=activ)
            # I add another 1x1 convolution in img disc
            _layers = [conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1, activation=activ),
                       nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)]   # 4
            self.img_disc_128 = nn.Sequential(*_layers)
            self.max_out_size = 128
            
        if input_size > 128:
            self.img_encoder_256  = shareImageDown(256, num_chan, enc_dim, norm, shared_block)  # 8
            self.pair_disc_256  = DiscClassifier(enc_dim, emb_dim, feat_size=4,  norm=norm, activ=activ)
            _layers = [conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1, activation=activ),
                       nn.Conv2d(enc_dim, 1, kernel_size=4, padding = 0, bias=True)]   # 4
            self.img_disc_256 = nn.Sequential(*_layers)
            self.max_out_size = 256
        
        self.apply(weights_init)
        print ('>> initialized a {} size discriminator'.format(input_size))

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
        assert self.max_out_size >= img_size, 'image size {} exceeds expected maximum size {}'.format(img_size, self.max_out_size)

        img_encoder_sym  = 'img_encoder_{}'.format(img_size)
        img_disc_sym = 'img_disc_{}'.format(img_size)
        pair_disc_sym = 'pair_disc_{}'.format(img_size)

        img_encoder = getattr(self, img_encoder_sym)
        img_disc    = getattr(self, img_disc_sym)
        pair_disc   = getattr(self, pair_disc_sym)

        sent_code = self.context_emb_pipe(embdding)
        
        img_code = img_encoder(images)
        pair_disc_out = pair_disc(sent_code, img_code)
        img_disc_out  = img_disc(img_code)
        
        out_dict['pair_disc']     = pair_disc_out
        out_dict['img_disc']      = img_disc_out
        out_dict['content_code']  = img_code # useless
        return out_dict