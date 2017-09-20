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
    # helper function
    def plot_imgs(samples, epoch, typ, name, path=''):

        tmpX = save_images(samples, save=not path == '', save_path=os.path.join(path, '{}_epoch{}_{}.png'.format(name, epoch, typ)), dim_ordering='th')
        plot_img(X=tmpX, win='{}_{}.png'.format(name, typ), env=mode_name)

    def fake_sampler(bz, n ):
        x = {}
        x['output_64'] = np.random.rand(bz, 3, 64, 64)
        x['output_128'] = np.random.rand(bz, 3, 128, 128)
        x['output_256'] = np.random.rand(bz, 3, 256, 256)
        return x, x, np.random.rand(bz, 1024), None, None

    ngen = getattr(args, 'ngen', 1)
    decay_count = getattr(args, 'decay_count', 20000) # when this iteration, decay lr by 2
    use_content_loss = getattr(args, 'use_content_loss', False)
    d_lr = args.d_lr
    g_lr = args.g_lr
    tot_epoch = args.maxepoch

    train_sampler = dataset.train.next_batch
    test_sampler  = dataset.test.next_batch
    number_example = dataset.train._num_examples

    # train_sampler = fake_sampler
    # test_sampler = fake_sampler
    # number_example = 16

    updates_per_epoch = int(number_example / args.batch_size)
    num_test = 64 // args.batch_size # number of testing samples to show
    if args.wgan:
        optimizerD = optim.RMSprop(netD.parameters(), lr= d_lr,  weight_decay=args.weight_decay)
        optimizerG = optim.RMSprop(netG.parameters(), lr= g_lr,  weight_decay=args.weight_decay)
    else:  
        optimizerD = optim.Adam(netD.parameters(), lr= d_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        optimizerG = optim.Adam(netG.parameters(), lr= g_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay) 
        
    model_folder = os.path.join(model_root, mode_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        
    if args.reuse_weigths:
        D_weightspath = os.path.join(model_folder, 'D_epoch{}.pth'.format(args.load_from_epoch))
        G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
        if os.path.exists(D_weightspath):
            weights_dict = torch.load(D_weightspath,map_location=lambda storage, loc: storage)
            netD.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(D_weightspath))

        if os.path.exists(G_weightspath):
            weights_dict = torch.load(G_weightspath,map_location=lambda storage, loc: storage)
            netG.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(G_weightspath))
        start_epoch = args.load_from_epoch
    else:
        start_epoch = 1

    d_loss_plot = plot_scalar(name = "d_loss", env= mode_name, rate = args.display_freq)
    g_loss_plot = plot_scalar(name = "g_loss", env= mode_name, rate = args.display_freq)
    content_loss_plot = plot_scalar(name = "content_loss", env= mode_name, rate = args.display_freq)
    lr_plot = plot_scalar(name = "lr", env= mode_name, rate = args.display_freq)

    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z, netG.device_id, requires_grad=False)

    z_test = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z_test = to_device(z_test, netG.device_id, volatile=True)    

    global_iter = 0
    for epoch in range(start_epoch, tot_epoch):
        
        # learning rate
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2
            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)
        
        for it in range(updates_per_epoch):
           
            ''' Sample data '''
            images, wrong_images, embeddings, _, _ = train_sampler(args.batch_size, args.num_emb)
            embeddings = to_device(embeddings, netD.device_id, requires_grad=False)
            z.data.normal_(0, 1)
            
            ''' update D '''        
            for p in netD.parameters(): p.requires_grad = True
            netD.zero_grad()

            g_emb = Variable(embeddings.data, volatile=False)
            fake_images, _ = netG(g_emb, z) # forward generator

            discriminator_loss = 0
            for key, _ in fake_images.items():
                # iterate over image of different sizes.
                this_img   = to_device(images[key], netD.device_id)
                this_wrong = to_device(wrong_images[key], netD.device_id)
                this_fake  = Variable(fake_images[key].data) # to cut connection to netG

                # TODO do we need multiple embeddings?
                # A faster implementation it is ok right?
                # joint_inputs = torch.cat([this_img, this_wrong, this_fake], 0)
                # joint_embeddings = torch.cat([embeddings, embeddings, embeddings], 0)
                # all_dict = netD(joint_inputs)
                # bz = joint_inputs.size(0) / 3 # the size for each type [this_img, this_wrong, this_fake]

                # real_logit, real_img_logit  =  real_dict['pair_disc'][:bz], real_dict['img_disc'][:bz]
                # wrong_logit, wrong_img_logit =  wrong_dict['pair_disc'][bz:bz*2], wrong_dict['img_disc'][bz:bz*2]
                # fake_logit, fake_img_logit =  fake_dict['pair_disc'][bz*2:], fake_dict['img_disc'][bz*2:]

                real_dict   = netD(this_img,   embeddings)
                wrong_dict  = netD(this_wrong, embeddings)
                fake_dict   = netD(this_fake,  embeddings)
                real_logit, real_img_logit  =  real_dict['pair_disc'], real_dict['img_disc']
                wrong_logit, wrong_img_logit =  wrong_dict['pair_disc'], wrong_dict['img_disc']
                fake_logit, fake_img_logit =  fake_dict['pair_disc'], fake_dict['img_disc']

                # compute loss
                chose_img_real = wrong_img_logit if random.random() > 0.5 else real_img_logit
                discriminator_loss += compute_d_pair_loss(real_logit, wrong_logit, fake_logit, args.wgan )
                discriminator_loss += compute_d_img_loss(chose_img_real, fake_img_logit, args.wgan )
        
            d_loss_val  = discriminator_loss.cpu().data.numpy().mean()
            d_loss_val = -d_loss_val if args.wgan else d_loss_val
            discriminator_loss.backward()
            optimizerD.step()    
            netD.zero_grad()
            d_loss_plot.plot(d_loss_val)

            ''' update G '''
            for p in netD.parameters(): 
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            z.data.normal_(0, 1) # resample random noises
            fake_images, kl_loss = netG(embeddings, z)
            
            loss_val  = 0
            generator_loss = args.KL_COE*kl_loss
            for key, _ in fake_images.items():
                # iterate over image of different sizes.
                this_fake  = fake_images[key]
                fake_dict  = netD(this_fake,  embeddings)
                fake_pair_logit, fake_img_logit, fake_img_code  = \
                fake_dict['pair_disc'], fake_dict['img_disc'], fake_dict['content_code']
                generator_loss += compute_g_loss(fake_pair_logit, fake_img_logit, args.wgan)               

            generator_loss.backward()
            g_loss_val = generator_loss.cpu().data.numpy().mean()
            
            optimizerG.step()    
            netG.zero_grad()
            g_loss_plot.plot(g_loss_val)
            lr_plot.plot(g_lr)

            global_iter += 1

            # visualize train samples
            if it % 50 == 0:
                for k, sample in fake_images.items():
                    plot_imgs(sample.cpu().data.numpy(), epoch, k, 'train_samples')
                    plot_imgs(images[k], epoch, k, 'train_images')
                print ('[epoch %d/%d iter %d]: lr = %.6f g_loss = %.5f d_loss= %.5f' % (epoch, tot_epoch, it, g_lr, g_loss_val, d_loss_val))

        ''' visualize test per epoch '''
        # generate samples
        gen_samples = []
        img_samples = []
        for idx_test in range(num_test):
            #sent_emb_test, _ =  netG.condEmbedding(test_embeddings)
            test_images, _, test_embeddings, _, _ = test_sampler(args.batch_size, 1)
            test_embeddings = to_device(test_embeddings, netG.device_id, requires_grad=False)
            z.data.normal_(0, 1)
            samples, _ = netG(test_embeddings, z)
            gen_samples.append(samples)
            img_samples.append(test_images)

        # visualize samples
        types = gen_samples[0].keys()
        for t in types:
            tmp = []
            tmp2 = []
            for i in range(len(gen_samples)):
                tmp.append(gen_samples[i][t].cpu().data.numpy())
                tmp2.append(img_samples[i][t])
                
            tmp_gen_samples = np.concatenate(tmp, axis=0)
            tmp_img_samples = np.concatenate(tmp2, axis=0)
            plot_imgs(tmp_gen_samples, epoch, t, 'test_samples', path=model_folder)
            plot_imgs(tmp_img_samples, epoch, t, 'test_images', path=model_folder)
    
        # save weights      
        if epoch % args.save_freq == 0:
            torch.save(netD.state_dict(), os.path.join(model_folder, 'D_epoch{}.pth'.format(epoch)))
            torch.save(netG.state_dict(), os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch)))
            print('save weights at {}'.format(model_folder))
        
        print ('epoch {}/{} finished ...'.format(epoch, tot_epoch))