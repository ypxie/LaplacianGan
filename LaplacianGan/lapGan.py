import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter

from torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
from .proj_utils.torch_utils import *
import time, json

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

def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_logit, prob=0.5, wgan=False):
    if wgan:
        dloss = (prob*wrong_img_logit + (1-prob)*real_img_logit) - real_logit 
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
        
        wrong_d_loss = torch.mean( ((wrong_img_logit) -1)**2)
        real_d_loss = torch.mean( ((real_img_logit) -1)**2)

        real_img_d_loss = wrong_d_loss * prob + real_d_loss * (1-prob)
        fake_d_loss =  torch.mean( ((fake_logit))**2)
        return fake_d_loss + real_img_d_loss

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

def load_partial_state_dict(model, state_dict):

        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                          name, own_state[name].size(), param.size()))
                raise
        print ('>> load partial state dict: {} initialized'.format(len(state_dict)))

def inter_across(embeddings):
    # embeddings (b, dim)
    B, _ = embeddings.shape
    res = embeddings.copy()
    for idx in range(B):
        ridx = random.randint(0, B-1)
        res[idx] = 0.5*(embeddings[idx], embeddings[ridx])
    return res
    
def train_gans(dataset, model_root, mode_name, netG, netD, args):
    print('Using lapGan')
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
    if not args.debug_mode:
        train_sampler = dataset.train.next_batch
        test_sampler  = dataset.test.next_batch
        number_example = dataset.train._num_examples
        updates_per_epoch = int(number_example / args.batch_size)
    
    else:
        train_sampler = fake_sampler
        test_sampler = fake_sampler
        number_example = 16
        updates_per_epoch = 10

    ''' configure optimizer '''
    num_test_forward = 1 # 64 // args.batch_size // args.test_sample_num # number of testing samples to show
    if args.wgan:
        optimizerD = optim.RMSprop(netD.parameters(), lr= d_lr,  weight_decay=args.weight_decay)
        optimizerG = optim.RMSprop(netG.parameters(), lr= g_lr,  weight_decay=args.weight_decay)
    else:  
        optimizerD = optim.Adam(netD.parameters(), lr= d_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        optimizerG = optim.Adam(netG.parameters(), lr= g_lr, betas=(0.5, 0.9), weight_decay=args.weight_decay) 
        
    model_folder = os.path.join(model_root, mode_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    plot_save_path = os.path.join(model_folder, 'plot_save.pth')
    plot_dict = {'disc':[], 'gen':[]}

    ''' load model '''
    if args.reuse_weigths :
        D_weightspath = os.path.join(model_folder, 'D_epoch{}.pth'.format(args.load_from_epoch))
        G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))

        if os.path.exists(D_weightspath) and os.path.exists(G_weightspath):
            
            #assert os.path.exists(D_weightspath) and os.path.exists(G_weightspath)
            weights_dict = torch.load(D_weightspath, map_location=lambda storage, loc: storage)
            # !! force load by renaming
            # import pdb; pdb.set_trace()
            print('reload weights from {}'.format(D_weightspath))
            # weights_dict_copy = {}
            # for idx, k1 in enumerate(weights_dict.keys()):
            #     # only iteraate weights_dict because netD may has more layers for multiple resolutions
            #     k2 = netD.state_dict().keys()[idx]
            #     weights_dict_copy[k2] = weights_dict[k1]
            load_partial_state_dict(netD, weights_dict)
            # netD.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(G_weightspath))
            weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
            load_partial_state_dict(netG, weights_dict)
            # netG.load_state_dict(weights_dict)# 12)

            start_epoch = args.load_from_epoch + 1
            if os.path.exists(plot_save_path):
                plot_dict = torch.load(plot_save_path)
        else:
            print('{} and {} do not exist!!'.format(D_weightspath, G_weightspath))
            start_epoch = 1
    else:
        start_epoch = 1

    d_loss_plot = plot_scalar(name = "d_loss", env= mode_name, rate = args.display_freq)
    g_loss_plot = plot_scalar(name = "g_loss", env= mode_name, rate = args.display_freq)
    content_loss_plot = plot_scalar(name = "content_loss", env= mode_name, rate = args.display_freq)
    lr_plot = plot_scalar(name = "lr", env= mode_name, rate = args.display_freq)

    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z, netG.device_id, requires_grad=False)
    # test the fixed image for every epoch
    fixed_images, _, fixed_embeddings, _, _ = test_sampler(args.batch_size, 1)
    fixed_embeddings = to_device(fixed_embeddings, netG.device_id, volatile=True)
    fixed_z_data = [torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1) for _ in range(args.test_sample_num)]
    fixed_z_list = [to_device(a, netG.device_id, volatile=True) for a in fixed_z_data] # what?


    # z_test = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    # z_test = to_device(z_test, netG.device_id, volatile=True)    

    global_iter = 0
    gen_iterations = 0
    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        # learning rate
        if epoch % args.epoch_decay == 0:
            d_lr = min(d_lr/2, 0.00005)
            g_lr = min(g_lr/2, 0.00005)
            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)
        

        for it in range(updates_per_epoch):
            
            netG.train()
            if gen_iterations < 100 or gen_iterations % 100 == 0:
                ncritic = 25
            else:
                ncritic = args.ncritic

            for _ in range(args.ncritic):
                ''' Sample data '''
                images, wrong_images, np_embeddings, _, _ = train_sampler(args.batch_size, args.num_emb)
                embeddings = to_device(np_embeddings, netD.device_id, requires_grad=False)
                z.data.normal_(0, 1)
                
                ''' update D '''        
                for p in netD.parameters(): p.requires_grad = True
                netD.zero_grad()
                
                g_emb = Variable(embeddings.data, volatile=True)
                g_z = Variable(z.data , volatile=True)
                # forward generator
                fake_images, _ = netG(g_emb, g_z) 
                
                discriminator_loss = 0
                d_loss_val_dict = {}
                for key, _ in fake_images.items():
                    # iterate over image of different sizes.
                    this_img   = to_device(images[key], netD.device_id)
                    this_wrong = to_device(wrong_images[key], netD.device_id)
                    this_fake  = Variable(fake_images[key].data) # to cut connection to netG

                    real_dict   = netD(this_img,   embeddings)
                    wrong_dict  = netD(this_wrong, embeddings)
                    fake_dict   = netD(this_fake,  embeddings)
                    real_logit, real_img_logit   =  real_dict['pair_disc'], real_dict['img_disc']
                    wrong_logit, wrong_img_logit =  wrong_dict['pair_disc'], wrong_dict['img_disc']
                    fake_logit, fake_img_logit   =  fake_dict['pair_disc'], fake_dict['img_disc']

                    # compute loss
                    #chose_img_real = wrong_img_logit if random.random() > 0.1 else real_img_logit
                    discriminator_loss += compute_d_pair_loss(real_logit, wrong_logit, fake_logit, args.wgan)
                    discriminator_loss += compute_d_img_loss(wrong_img_logit, real_img_logit, fake_img_logit, prob=0.5, wgan=args.wgan ) 

                d_loss_val  = discriminator_loss.cpu().data.numpy().mean()
                d_loss_val = -d_loss_val if args.wgan else d_loss_val
                discriminator_loss.backward()
                optimizerD.step()    
                netD.zero_grad()
                d_loss_plot.plot(d_loss_val)
                plot_dict['disc'].append(d_loss_val)
        
            ''' update G '''
            for p in netD.parameters(): 
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            #_, _, embeddings, _, _ = train_sampler(args.batch_size, args.num_emb)
            ''' Interpolate across samples '''
            # if args.emb_interp:
            #     np_embeddings = inter_across(np_embeddings)
            #     embeddings = to_device(np_embeddings, netD.device_id, requires_grad=False)

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
                
                if use_content_loss:
                    if epoch >= 50:
                        this_img   = to_device(images[key], netD.device_id)
                        real_dict   = netD(this_img,   embeddings)
                        real_img_code = real_dict['content_code']
                        #l2 = torch.mean((real_img_code - fake_img_code)**2) 
                        #conten_loss = GaussianLogDensity(real_img_code, fake_img_code)
                        conten_loss = torch.mean(torch.abs(fake_img_code - real_img_code))* 4
                        generator_loss += conten_loss
                        content_loss_plot.plot(conten_loss.cpu().data.numpy().mean())

            generator_loss.backward()
            g_loss_val = generator_loss.cpu().data.numpy().mean()
            
            optimizerG.step()    
            netG.zero_grad()
            g_loss_plot.plot(g_loss_val)
            lr_plot.plot(g_lr)
            plot_dict['gen'].append(g_loss_val)
            gen_iterations += 1
            global_iter += 1
            # visualize train samples
            if it % 50 == 0:
                for k, sample in fake_images.items():
                    # plot_imgs(sample.cpu().data.numpy(), epoch, k, 'train_samples')
                    plot_imgs([images[k], sample.cpu().data.numpy()], epoch, k, 'train_images')
                print ('[epoch %d/%d iter %d]: lr = %.6f g_loss = %.5f d_loss= %.5f' % (epoch, tot_epoch, it, g_lr, g_loss_val, d_loss_val))

        ''' visualize test per epoch '''
        # generate samples
        gen_samples = []
        img_samples = []
        vis_samples = {}
        for idx_test in range(num_test_forward + 1):
            #sent_emb_test, _ =  netG.condEmbedding(test_embeddings)
            if idx_test == 0:
                test_images, test_embeddings = fixed_images, fixed_embeddings

            else:
                test_images, _, test_embeddings, _, _ = test_sampler(args.batch_size, 1)
                test_embeddings = to_device(test_embeddings, netG.device_id, volatile=True)
                testing_z = Variable(z.data, volatile=True)

            tmp_samples = {}

            for t in range(args.test_sample_num):
                
                if idx_test == 0: # plot fixed
                    testing_z = fixed_z_list[t]
                else:
                    testing_z.data.normal_(0, 1)
                
                samples, _ = netG(test_embeddings, testing_z)
                
                if idx_test == 0 and t == 0:  
                    for k in samples.keys():
                        vis_samples[k] = [None for i in range(args.test_sample_num + 1)] # +1 to fill real image
                
                # Oops! very tricky to organize data for plot inputs!!!
                # vis_samples[k] = [real data, sample1, sample2, sample3, ... sample_args.test_sample_num]
                for k, v in samples.items():
                    cpu_data = v.cpu().data.numpy()
                    if t == 0:
                        if vis_samples[k][0] == None:
                            vis_samples[k][0] = test_images[k]
                        else:
                            vis_samples[k][0] =  np.concatenate([ vis_samples[k][0], test_images[k]], 0) 

                    if vis_samples[k][t+1] == None:
                        vis_samples[k][t+1] = cpu_data
                    else:
                        vis_samples[k][t+1] = np.concatenate([vis_samples[k][t+1], cpu_data], 0)

        end_timer = time.time() - start_timer
        # visualize samples
        for typ, v in vis_samples.items():
            plot_imgs(v, epoch, typ, 'test_samples', path=model_folder)
    
        # save weights      
        if epoch % args.save_freq == 0:
            torch.save(netD.state_dict(), os.path.join(model_folder, 'D_epoch{}.pth'.format(epoch)))
            torch.save(netG.state_dict(), os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch)))
            print('save weights at {}'.format(model_folder))
            torch.save(plot_dict, plot_save_path)


        print ('epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))