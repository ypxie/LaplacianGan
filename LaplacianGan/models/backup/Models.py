import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict
from ..proj_utils.model_utils import *

class condEmbedding(nn.Module):
    def __init__(self, noise_dim, emb_dim):
        super(condEmbedding, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.noise_dim = noise_dim
        self.emb_dim = emb_dim
        self.linear  = nn.Linear(noise_dim, emb_dim*2)
        
    def forward(self, inputs):
        '''
        inputs: (B, dim)
        return: mean (B, dim), logsigma (B, dim)
        '''
        out = F.selu( self.linear(inputs) )
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]
        return mean, log_sigma

# DCGAN generator
def brach_out(feat_dim, out_dim, norm, repeat=2):
    _layers = []
    for _ in range(repeat):
        _layers += [Bottleneck(feat_dim, feat_dim) ] 
    
    _layers += [nn.Conv2d(feat_dim,  out_dim, 
                kernel_size = 3, padding=1, bias=True)]    
    _layers += [torch.nn.Tanh()]
    return _layers

def get_internal(feat_dim, out_dim, norm, repeat=1):
    _layers = []
    _layers = [nn.ConvTranspose2d(feat_dim, out_dim , 
                   kernel_size=4,stride=2, padding=1, bias=False)]
    _layers += [nn.SELU(inplace=False)]
    _layers += [getNormLayer(norm)(out_dim )]
    for _ in range(repeat):
        _layers += [nn.Conv2d(out_dim,  out_dim, 
                    kernel_size = 3, padding=1, bias=True)]
        _layers += [nn.SELU(inplace=False)]
        _layers += [getNormLayer(norm)(out_dim )]

    return _layers

class Generator(nn.Module):
    def __init__(self, input_size, noise_dim, num_chan, 
                 emb_dim, hid_dim, norm = 'ln',branch=True):
        super(Generator, self).__init__()
        self.__dict__.update(locals())
        self.register_buffer('device_id', torch.zeros(1))
        self.condEmbedding = condEmbedding(noise_dim, emb_dim)
        # We need to know how many layers we will use at the beginning
        mult = 4
        node_dict = OrderedDict()
        feat_tuple_dict = OrderedDict()

        feat_size = 4

        _layers = [ nn.ConvTranspose2d(emb_dim, hid_dim * mult, \
               kernel_size=4,  stride=1, padding=0, bias=False) ]
        _layers += [nn.SELU(inplace=False)] 
        self.node_4 = nn.Sequential(*_layers)
        
        this_mult = max(mult//2, 1)
        _layers = get_internal(hid_dim * mult,  hid_dim * this_mult, norm, repeat=1)
        self.node_8 = nn.Sequential(*_layers)
        feat_size = feat_size*2
        feat_tuple_dict["map_8"] =  [hid_dim * this_mult , feat_size]
        mult = this_mult
        
        this_mult = max(mult//2, 1)
        _layers = get_internal(hid_dim * mult,  hid_dim * this_mult, norm, repeat=1)
        self.node_16 = nn.Sequential(*_layers)
        feat_size = feat_size*2
        feat_tuple_dict["map_16"] =  [hid_dim * this_mult , feat_size]
        mult = this_mult
        

        this_mult = max(mult//2, 1)
        _layers = get_internal(hid_dim * mult,  hid_dim * this_mult, norm, repeat=1)
        self.node_32 = nn.Sequential(*_layers)
        feat_size = feat_size*2
        feat_tuple_dict["map_32"] =  [hid_dim * this_mult , feat_size]
        mult = this_mult
        
        if input_size == 256:
            this_mult = max(mult//2, 1)
            _layers = get_internal(hid_dim * mult,  hid_dim * this_mult, norm, repeat=1)
            self.node_64 = nn.Sequential(*_layers)
            feat_size = feat_size*2
            feat_tuple_dict["map_64"] =  [hid_dim * this_mult , feat_size]
            mult = this_mult
            
            this_mult = max(mult//2, 1)
            _layers = get_internal(hid_dim * mult,  hid_dim * this_mult, norm, repeat=1)
            self.node_128 = nn.Sequential(*_layers)
            feat_size = feat_size*2
            feat_tuple_dict["map_128"] =  [hid_dim * this_mult , feat_size]
            mult = this_mult

        # To the final layer
        _layers = [torch.nn.ConvTranspose2d(hid_dim * this_mult, hid_dim, kernel_size=4, 
                   stride=2, padding=1, bias=False)]
        _layers += [nn.SELU(inplace=False)]
        _layers += [nn.Conv2d(hid_dim, num_chan, 
                    kernel_size = 3, padding=1, bias=True)]
        _layers += [torch.nn.Tanh()]

        feat_size *= 2
        self.node_final = nn.Sequential(*_layers)
        
        # start from here, brach out to 64, 128 for multi scale
        # feat_tuple_list = list(feat_tuple_dict.items())
        # for map_key, feat_tuple in feat_tuple_list[-4:-3]: 
        #     # we only consider, for example(32, 64) if final size is 256
        #     feat_dim, this_size = feat_tuple
        #     print("branch_{}", feat_dim, this_size)
        #     _layers = brach_out(feat_dim, 3, norm, repeat=2)
        #     node_dict["branch_{}".format(this_size)] = nn.Sequential(*_layers)
        
        if branch:
            feat_dim, this_size = feat_tuple_dict["map_64"]
            _layers = brach_out(feat_dim, 3, norm, repeat=2)
            self.branch_64 = nn.Sequential(*_layers)

        # node_dict should have (node_4, 8, 16, 32, 64, 128, 256, brach 64, _128)
        for key, val in node_dict.items():
            setattr(self, key, val)

        self.apply(weights_init)

    def forward(self, inputs):
        out_dict = OrderedDict()
        inputs = inputs.view(inputs.size()[0], self.emb_dim, 1, 1)
        node_4   = self.node_4(inputs)
        node_8   = self.node_8(node_4)
        node_16  = self.node_16(node_8)
        node_32  = self.node_32(node_16)
        if self.input_size == 256:
            node_64  = self.node_64(node_32)
            node_128 = self.node_128(node_64)
            final_output = self.node_final(node_128)
        else:
            node_64  = self.node_final(node_32)
            final_output = node_64

        out_dict['output_final']    = final_output
        if self.branch:
            branch_64 = self.branch_64(node_64)
            out_dict['branch_first']  = branch_64
        #branch_128 = self.branch_128(node_128)

        return out_dict

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
        mult = 1

        _layers = [nn.Conv2d(num_chan, hid_dim, 
                    kernel_size = 3, padding=1, bias=True)]
        _layers += [nn.LeakyReLU(0.2)]
        
        input_dim = hid_dim
        while feat_size > 4:
            
            this_mult = mult*2
            this_out_dim = min(this_mult*hid_dim, out_dim)
            #print('this feat_size {}, this_out_dim {}'.format(feat_size, this_out_dim))
            _layers += [nn.Conv2d(input_dim,  this_out_dim, kernel_size=4, 
                        stride=2, padding=1, bias=True)]
            _layers += [getNormLayer(norm)(this_out_dim )]
            _layers += [nn.LeakyReLU(0.2)]
            #_layers += [Bottleneck(this_out_dim, this_out_dim) ]
            mult = mult*2
            input_dim = this_out_dim

            feat_size  = feat_size//2

        if this_out_dim != out_dim:
            _layers += [nn.Conv2d(this_out_dim, out_dim, 
                       kernel_size=1, stride=1, padding=0, bias=True)]
            _layers += [nn.LeakyReLU(0.2)]

        self.main = nn.Sequential(*_layers)
        
    def forward(self, inputs):
        # inputs (B, C, H, W), must be dividable by 32
        # return (B, C, 4, 4)
        output = self.main(inputs)
        return output

class Discriminator(torch.nn.Module):
    '''
    enc_dim: Reduce images inputs to (B, enc_dim, H, W)
    emb_dim: The sentence embedding dimension.
    '''
    def __init__(self, input_size, num_chan,  hid_dim, enc_dim, emb_dim, norm='ln'):
        super(Discriminator, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.__dict__.update(locals())
        
        _layers = [nn.Linear(emb_dim, emb_dim)]
        _layers += [nn.LeakyReLU(0.2, inplace=False)]
        self.context_emb_pipe = nn.Sequential(*_layers)
        self.img_encoder = ImageEncoder(input_size, num_chan, hid_dim, enc_dim, norm=norm)
        
        # it is ugly to hard written,but to share weights between them.
        self.img_encoder_64 = ImageEncoder(64, num_chan, hid_dim, enc_dim, norm=norm)

        composed_dim = enc_dim + emb_dim

        _layers = [nn.Conv2d(composed_dim, emb_dim, 
                    kernel_size = 1, padding = 0, bias=True)]
        _layers += [nn.SELU(inplace=False)]
        #_layers += [Bottleneck(emb_dim, emb_dim, kernel_size=1)]
        _layers += [getNormLayer(norm)(emb_dim )]
        _layers += [nn.Conv2d(emb_dim, 1, 
                    kernel_size = 4, padding=0, bias=True)]
        
        self.final_stage = nn.Sequential(*_layers)

    def encode_img(self, images):
        img_size = images.size()[3]
        #print('images size ', images.size())
        if img_size == 64:
            img_code = self.img_encoder_64(images)
        else:    
            img_code = self.img_encoder(images)
        return img_code

    def forward(self, images, c_var):
        '''
        images: (B, C, H, W)
        c_var : (B, emb_dim)
        outptu: (B, 1)
        '''
        img_size = images.size()[3]
        
        img_code  = self.encode_img(images)
        sent_code = self.context_emb_pipe(c_var)
        
        sent_code =  sent_code.unsqueeze(-1).unsqueeze(-1)
        dst_shape = list(sent_code.size())
        dst_shape[2] =  img_code.size()[2] 
        dst_shape[3] =  img_code.size()[3] 

        sent_code = sent_code.expand(dst_shape)
    
        compose_input = torch.cat([img_code, sent_code], dim=1)
        #print('compose_input size ', compose_input.size())
        
        output = self.final_stage(compose_input)
        #print('output shape is ', output.size())
        b, outdim = output.size()[0:2]
        
        return output.view(b, outdim)

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
