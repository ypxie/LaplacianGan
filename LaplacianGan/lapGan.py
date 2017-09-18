import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
from .proj_utils.torch_utils import *

TINY = 1e-8


def compute_d_pair_loss(real_logit, wrong_logit, fake_logit, wgan=False):
    if wgan:
        disc = wrong_logit  + fake_logit - 2*real_logit
        return torch.mean(disc)
    else:    
        # ones_target  =  Variable(real_logit.data.new(real_logit.size()).fill_(1.0), requires_grad=False)
        # zeros_target =  Variable(real_logit.data.new(real_logit.size()).fill_(0.0), requires_grad=False)

        # real_d_loss =\
        #     F.binary_cross_entropy_with_logits( real_logit,
        #                                         ones_target)
        # real_d_loss = torch.mean(real_d_loss)

        # wrong_d_loss =\
        #     F.binary_cross_entropy_with_logits( wrong_logit,
        #                                         zeros_target)
        # wrong_d_loss = torch.mean(wrong_d_loss)

        # fake_d_loss =\
        #     F.binary_cross_entropy_with_logits( fake_logit,
        #                                         zeros_target)
        # fake_d_loss = torch.mean(fake_d_loss)
        
        real_d_loss  = torch.mean( ((real_logit) -1)**2)
        wrong_d_loss = torch.mean( ((wrong_logit))**2)
        fake_d_loss  = torch.mean( ((fake_logit))**2)
        
        discriminator_loss =\
            real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        return discriminator_loss

def compute_d_img_loss(real_logit, fake_logit, wgan=False):
    if wgan:
        dloss = fake_logit - real_logit 
        return torch.mean(dloss)
    else:
        # ones_target  =  Variable(real_logit.data.new(real_logit.size()).fill_(1.0), 
        #                         requires_grad=False)
        # zeros_target =  Variable(real_logit.data.new(real_logit.size()).fill_(0.0), 
        #                         requires_grad=False)
        # real_d_loss =  F.binary_cross_entropy_with_logits( real_logit, ones_target)
        # real_d_loss = torch.mean(real_d_loss)
        # fake_d_loss =  F.binary_cross_entropy_with_logits(fake_logit, zeros_target)
        # fake_d_loss =  torch.mean(fake_d_loss) 
        
        real_d_loss =  torch.mean( ((real_logit) -1)**2)
        fake_d_loss =  torch.mean( ((fake_logit))**2)
        return fake_d_loss + real_d_loss

def compute_g_loss(fake_pair_logit, fake_img_logit, wgan=False):
    if wgan:
        gloss = -fake_pair_logit - fake_img_logit
        #gloss = -fake_img_logit
        return torch.mean(gloss)
    else:      
        # ones_target_pair  =  Variable(fake_pair_logit.data.new(fake_pair_logit.size()).
        #                        fill_(1.0), requires_grad=False)

        # ones_target_img  =  Variable(fake_img_logit.data.new(fake_img_logit.size()).
        #                        fill_(1.0), requires_grad=False)

        # generator_loss = torch.mean(F.binary_cross_entropy_with_logits(fake_pair_logit, ones_target_pair) )\
        #               + torch.mean(F.binary_cross_entropy_with_logits(fake_img_logit,  ones_target_img) )
        
        generator_loss = torch.mean( ((fake_pair_logit) -1)**2 ) + \
                         torch.mean( ((fake_img_logit)  -1)**2 )
        return generator_loss

def GaussianLogDensity(x, mu, log_var = 'I'):
    # x: real mean, mu: fake mean
    if log_var is  'I':
        log_var = Variable(mu.data.new(mu.size()).fill_(0.0), requires_grad=False)
    
    c = Variable(mu.data.new(1).fill_(2*3.14159265359), requires_grad=False)
    var = torch.exp(log_var)
    x_mu2 = (x - mu)**2   # [Issue] not sure the dim works or not?
    #print(x_mu2.size(), var.size(), (var + TINY).size())
    x_mu2_over_var = x_mu2 / (var + TINY)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = -torch.mean(torch.mean(log_prob, -1) )   # keep_dims=True,
    return log_prob

def train_gans(dataset, model_root, mode_name, netG, netD, args):
    ngen = getattr(args, 'ngen', 1)
    decay_count = getattr(args, 'decay_count', 20000) # when this iteration, decay lr by 2
    use_content_loss = getattr(args, 'use_content_loss', False)
    d_lr = args.d_lr
    g_lr = args.g_lr

    train_sampler = dataset.train.next_batch
    test_sampler  = dataset.test.next_batch
    num_test = 2 # number of testing samples to show
    if args.wgan:
        optimizerD = optim.RMSprop(netD.parameters(), lr= args.d_lr,  weight_decay=args.weight_decay)
        optimizerG = optim.RMSprop(netG.parameters(), lr= args.g_lr,  weight_decay=args.weight_decay)
    else:  
        optimizerD = optim.Adam(netD.parameters(), lr= args.d_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        optimizerG = optim.Adam(netG.parameters(), lr= args.g_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay) 
        
    model_folder = os.path.join(model_root, mode_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        
    D_weightspath = os.path.join(model_folder, 'd_weights.pth')
    G_weightspath = os.path.join(model_folder, 'g_weights.pth')
    if args.reuse_weigths == 1:
        if os.path.exists(D_weightspath):
            weights_dict = torch.load(D_weightspath,map_location=lambda storage, loc: storage)
            netD.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(D_weightspath))
        
        if os.path.exists(G_weightspath):
            weights_dict = torch.load(G_weightspath,map_location=lambda storage, loc: storage)
            netG.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(G_weightspath))
    d_loss_plot = plot_scalar(name = "d_loss", env= mode_name, rate = args.display_freq)
    g_loss_plot = plot_scalar(name = "g_loss", env= mode_name, rate = args.display_freq)
    content_loss_plot = plot_scalar(name = "content_loss", env= mode_name, rate = args.display_freq)
    
    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z, netG.device_id, requires_grad=False)

    z_test = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z_test = to_device(z_test, netG.device_id, volatile=True)

    test_images, _, test_embeddings,_, _ = test_sampler(args.batch_size, 1)
    test_embeddings = to_device(test_embeddings, netG.device_id, requires_grad=False)

    one = netD.device_id.new(1).fill_(1)  
    one_neg = one * -1
    
    l1_loss = nn.L1Loss()

    gen_iterations, disc_iterations = 0, 0
    for batch_count in range(args.maxepoch):
        
        if gen_iterations < 10 or gen_iterations % 100 == 0:
            ncritic = 10
        else:
            ncritic = args.ncritic
        
        if batch_count != 0 and batch_count%decay_count == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2
            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True
        for _ in range(ncritic):
            # (1) Update D network
            if args.wgan:
                for p in netD.parameters(): 
                    p.data.clamp_(-0.02, 0.02)

            # images should be dictionary. [64, final]
            images, wrong_images, embeddings, _, _ = train_sampler(args.batch_size, args.num_emb)
            
            embeddings = to_device(embeddings, netD.device_id, requires_grad=False)
            #sent_random, _ =  netG.condEmbedding(embeddings)    
            #sent_random =  Variable(sent_random.data, volatile=True)
            z.data.normal_(0, 1)
            #z = tf.random_normal([self.batch_size, cfg.Z_DIM])
            # fake_images dictionary, same order as images
            g_emb = Variable(embeddings.data, volatile=False)
            fake_images, _ = netG(g_emb, z)
            netD.zero_grad()

            discriminator_loss = 0
            for key, _ in fake_images.items():
                # iterate over image of different sizes.
                this_img   = to_device(images[key], netD.device_id)
                this_wrong = to_device(wrong_images[key], netD.device_id)

                this_fake  = Variable( fake_images[key].data ) # to cut connection to netG
                
                real_dict   = netD(this_img,   embeddings)
                wrong_dict  = netD(this_wrong, embeddings)
                fake_dict   = netD(this_fake,  embeddings)

                real_logit, real_img_logit  = \
                                 real_dict['pair_disc'], real_dict['img_disc']
                wrong_logit,wrong_img_logit = \
                                 wrong_dict['pair_disc'], wrong_dict['img_disc']
                fake_logit, fake_img_logit =  \
                                 fake_dict['pair_disc'], fake_dict['img_disc']

                chose_img_real = wrong_img_logit if random.random() > 0.5 else real_img_logit
                
                discriminator_loss += compute_d_pair_loss(real_logit,wrong_logit,
                                        fake_logit, args.wgan )
                discriminator_loss += compute_d_img_loss(chose_img_real, 
                                        fake_img_logit, args.wgan )
        
            d_loss_val  = discriminator_loss.cpu().data.numpy().mean()
            d_loss_val = -d_loss_val if args.wgan else d_loss_val
            discriminator_loss.backward()
            optimizerD.step()    
            netD.zero_grad()
            d_loss_plot.plot(d_loss_val)
            real_dict, wrong_dict, fake_dict = None, None, None

        # (2) Update G network
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        
        for _ in range(ngen):
            netG.zero_grad()
            images, wrong_images, embeddings, _, _ = train_sampler(args.batch_size, args.num_emb)
            embeddings = to_device(embeddings, netD.device_id, requires_grad=False)
            #sent_random, kl_loss =  netG.condEmbedding(embeddings, kl_loss=True)
            # fake_images dictionary, same order as images
            z.data.normal_(0, 1)
            fake_images, kl_loss = netG(embeddings, z)
            
            loss_val  = 0
            generator_loss = args.KL_COE*kl_loss
            for key, _ in fake_images.items():
                # iterate over image of different sizes.
                
                this_fake  = fake_images[key]
                fake_dict  = netD(this_fake,  embeddings)
                fake_pair_logit, fake_img_logit, fake_img_code  = \
                fake_dict['pair_disc'], fake_dict['img_disc'], fake_dict['content_code']
                
                if use_content_loss:
                    if batch_count > 1000:
                        this_img   = to_device(images[key], netD.device_id)
                        real_dict   = netD(this_img,   embeddings)
                        real_img_code = real_dict['content_code']
                        #l2 = torch.mean((real_img_code - fake_img_code)**2) 
                        #l1 =
                        #conten_loss = GaussianLogDensity(real_img_code, fake_img_code)
                        #conten_loss = torch.mean(torch.abs(fake_img_code - real_img_code))* 4
                        #generator_loss += conten_loss
                        content_loss_plot.plot(conten_loss.cpu().data.numpy().mean())

                generator_loss += compute_g_loss(fake_pair_logit, fake_img_logit, args.wgan)               
                
            generator_loss.backward()
            g_loss_val = generator_loss.cpu().data.numpy().mean()
            
            optimizerG.step()    
            netG.zero_grad()
            g_loss_plot.plot(g_loss_val)
            gen_iterations += 1
            fake_dict, real_dict, fake_images= None, None, None

        # Calculate dev loss and generate samples every 100 iters
        if batch_count % args.display_freq == 0:
            print('save tmp images, :)')
            #z1   = z_sampler(batch_size, args.noise_dim)
            for idx_test in range(num_test):
                #sent_emb_test, _ =  netG.condEmbedding(test_embeddings)
                z.data.normal_(0, 1)
                #c_test    =  Variable(c_test.data, volatile=True)
                samples, _ = netG(test_embeddings, z)
                for key, val in samples.items():
                    this_fake = val.cpu().data.numpy()
                    this_real = test_images[key]

                    fake_imgs = save_images(
                                this_fake,
                                os.path.join(args.save_folder,'samples_{}_{}.png'.format(key, batch_count) )
                                            ,save=False,dim_ordering = 'th'
                                )
                    #print(this_fake.shape)
                    plot_img(X=fake_imgs, win='sample_img_{}_{}'.format(key, idx_test), env=mode_name)
                    
                    if idx_test == 0: # for real, we only plot once
                        true_imgs = save_images(this_real, save=False,dim_ordering = 'th')
                        plot_img(X=true_imgs, win='real_img_{}'.format(key), env=mode_name)
            
        if batch_count % args.save_freq == 0:
            D_cur_weights = netD.state_dict()
            G_cur_weights = netG.state_dict()
            torch.save(D_cur_weights, D_weightspath)
            torch.save(G_cur_weights, G_weightspath)
            print('save weights to {} and {}'.format(D_weightspath, 
                   G_weightspath), batch_count,args.save_freq)

