import numpy as np
import os, sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.parallel

from torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
from .proj_utils.torch_utils import *
import time, json
from functools import partial
TINY = 1e-8

def to_variable(x, requires_grad=True,  var=True, volatile=False):
    
    if type(x) is Variable:
        return x
    if type(x) is np.ndarray:
        x = torch.from_numpy(x.astype(np.float32))
    if var:
        x = Variable(x, requires_grad=requires_grad, volatile=volatile)
    # x.volatile = volatile 
    
    return x

def to_device(src, var = True, volatile = False, requires_grad=True):
    requires_grad = requires_grad and (not volatile)
    src = to_variable(src, var=var, volatile=volatile, requires_grad=requires_grad)
    return src.cuda() 

def to_img_dict(*inputs):
    res = {}
   
    inputs = inputs[0]
    # if does not has tat scale image, we return a vector 
    if len(inputs[0].size()) != 1: 
        res['output_64'] = inputs[0]
    if len(inputs[1].size()) != 1: 
        res['output_128'] = inputs[1]
    if len(inputs[2].size()) != 1: 
        res['output_256'] = inputs[2]
    mean_var = (inputs[3], inputs[4])
    return res, mean_var

def get_KL_Loss(mu, logvar):
    # import pdb; pdb.set_trace()
    kld = mu.pow(2).add(logvar.mul(2).exp()).add(-1).mul(0.5).add(logvar.mul(-1))
    kl_loss = torch.mean(kld)
    return kl_loss

# def get_KL_Loss(mu, log_sigma):
#     loss = -log_sigma + .5 * (-1 + torch.exp(2. * log_sigma) + mu**2)
#     loss = torch.mean(loss)
#     return loss

def compute_d_pair_loss(real_logit, wrong_logit, fake_logit,  real_labels, fake_labels):

    criterion = nn.MSELoss()
   
    real_d_loss = criterion(real_logit, real_labels)
    wrong_d_loss = criterion(wrong_logit, fake_labels)
    fake_d_loss = criterion(fake_logit, fake_labels)

    # real_d_loss  = torch.mean( ((real_logit) -1)**2)
    # wrong_d_loss = torch.mean( ((wrong_logit))**2)
    # fake_d_loss  = torch.mean( ((fake_logit))**2)

    discriminator_loss =\
        real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
    return discriminator_loss

def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_img_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    wrong_d_loss = criterion(wrong_img_logit, real_labels)
    real_d_loss = criterion(real_img_logit, real_labels)
    fake_d_loss = criterion(fake_img_logit, fake_labels)

    return fake_d_loss + (wrong_d_loss+real_d_loss) / 2


def compute_g_loss(fake_logit, real_labels):

    criterion = nn.MSELoss()
    generator_loss = criterion(fake_logit, real_labels)
    return generator_loss

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

''' Note for data parallel
1. data_parallel devices list much be [0,N]
2. can not use any explicit operations for tensors (like, +, * -)
3. 
'''

def train_gans(dataset, model_root, mode_name, netG, netD, args, gpus):

    def data_parallel(func, ins):
            return func(ins[0], ins[1])

    use_img_loss = getattr(args, 'use_img_loss', True)

    print('>> using hd gan trainer')
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

    d_lr = args.d_lr
    g_lr = args.g_lr
    tot_epoch = args.maxepoch
    if not args.debug_mode:
        train_sampler  = iter(dataset.train)
        test_sampler   = iter(dataset.test)
        #number_example = len(dataset.train)
        #updates_per_epoch = int(number_example / args.batch_size)
        updates_per_epoch =  len(dataset.train) 
    else:
        train_sampler = fake_sampler
        test_sampler  = fake_sampler
        #number_example = 16
        updates_per_epoch = 10

    ''' configure optimizer '''
    num_test_forward = 1 # 64 // args.batch_size // args.test_sample_num # number of testing samples to show

    optimizerD = optim.Adam(netD.parameters(), lr= d_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr= g_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

    model_folder = os.path.join(model_root, mode_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    plot_save_path = os.path.join(model_folder, 'plot_save.pth')
    plot_dict = {'disc':[], 'gen':[]}

    ''' load model '''
    if args.reuse_weights :
        D_weightspath = os.path.join(model_folder, 'D_epoch{}.pth'.format(args.load_from_epoch))
        G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))

        if os.path.exists(D_weightspath) and os.path.exists(G_weightspath):

            #assert os.path.exists(D_weightspath) and os.path.exists(G_weightspath)
            weights_dict = torch.load(D_weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(D_weightspath))
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
            print ('{} or {} do not exist!!'.format(D_weightspath, G_weightspath))
            raise NotImplementedError
    else:
        start_epoch = 1

    d_loss_plot = plot_scalar(name = "d_loss", env= mode_name, rate = args.display_freq)
    g_loss_plot = plot_scalar(name = "g_loss", env= mode_name, rate = args.display_freq)
    content_loss_plot = plot_scalar(name = "content_loss", env= mode_name, rate = args.display_freq)
    lr_plot = plot_scalar(name = "lr", env=mode_name, rate = args.display_freq)
    kl_loss_plot = plot_scalar(name = "kl_loss", env= mode_name, rate = args.display_freq)

    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z)
    # test the fixed image for every epoch
    fixed_images, _, fixed_embeddings, _, _ = test_sampler.next()
    test_batch_size = fixed_embeddings.size(0)
    fixed_embeddings = to_device(fixed_embeddings)
    fixed_z_data = [torch.FloatTensor(test_batch_size, args.noise_dim).normal_(0, 1) for _ in range(args.test_sample_num)]
    fixed_z_list = [to_device(a) for a in fixed_z_data] # what?
    test_z = torch.FloatTensor(test_batch_size, args.noise_dim).normal_(0, 1)
    test_z = to_device(test_z)
    REAL_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(1)).cuda()
    FAKE_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(0)).cuda()
    # now assume the local is 5x5
    REAL_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 5).fill_(1)).cuda()
    FAKE_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 5).fill_(0)).cuda()

    def get_labels(logit):
        if logit.size(-1) == 1: 
            return REAL_global_LABELS.view_as(logit), FAKE_global_LABELS.view_as(logit)
        else:
            return REAL_local_LABELS.view_as(logit), FAKE_local_LABELS.view_as(logit)

    global_iter = 0
    gen_iterations = 0
    
    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        # learning rate
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2

            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)
        
        train_sampler = iter(dataset.train) # reset
        test_sampler  = iter(dataset.test)
        netG.train()
        netD.train()
        for it in range(updates_per_epoch):

            if epoch <= args.ncritic_epoch_range:
                if (epoch < 2) and (gen_iterations < 100 or (gen_iterations < 1000 and gen_iterations % 20 == 0))  :
                    ncritic = 5
                    #print ('>> set ncritic to {}'.format(ncritic)) 
                elif gen_iterations % 50 == 0:
                    ncritic = 10    
                else:
                    ncritic = args.ncritic
                   #print ('>> set ncritic to {}'.format(ncritic))
            else:
                ncritic = args.ncritic

            for _ in range(ncritic):
                ''' Sample data '''
                try:
                    images, wrong_images, np_embeddings, _, _ = train_sampler.next()
                except:
                    train_sampler = iter(dataset.train) # reset
                    images, wrong_images, np_embeddings, _, _ = train_sampler.next()
                    
                embeddings = to_device(np_embeddings) 
                z.data.normal_(0, 1)


                ''' update D '''
                for p in netD.parameters(): 
                    p.requires_grad = True
                netD.zero_grad()

                #g_emb = Variable(embeddings.data, volatile=True)
                #g_z = Variable(z.data , volatile=True)
                #fake_images, _ = to_img_dict(data_parallel(netG, (g_emb, g_z)))
                ## ''' Note that by setting this, we use different fake images in G and D updates '''
                fake_images, mean_var = to_img_dict(data_parallel(netG, (embeddings, z)))

                d_loss_val = 0
                discriminator_loss = 0
                image_loss_ratio = 1
                for key, _ in fake_images.items():
                    # iterate over image of different sizes.
                    this_img   = to_device(images[key]) 
                    this_wrong = to_device(wrong_images[key])
                    this_fake  = to_device(fake_images[key].data) # to cut connection to netG

                    real_logit, real_img_logit_global = data_parallel(netD, (this_img, embeddings))
                    wrong_logit, wrong_img_logit_global = data_parallel(netD, (this_wrong, embeddings))
                    fake_logit, fake_img_logit_global = data_parallel(netD, (this_fake,  embeddings))
                    # compute loss

                    real_labels, fake_labels = get_labels(real_logit)
                    pair_loss = compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

                    real_labels, fake_labels = get_labels(real_img_logit_global)
                    img_loss = compute_d_img_loss(wrong_img_logit_global, real_img_logit_global, fake_img_logit_global, real_labels, fake_labels)

                    discriminator_loss += (pair_loss + img_loss * image_loss_ratio)

                discriminator_loss.backward() # backward per discriminator (save memory)
                d_loss_val += discriminator_loss.cpu().data.numpy().mean()
                optimizerD.step()
                netD.zero_grad()

                d_loss_plot.plot(d_loss_val)
                plot_dict['disc'].append(d_loss_val)

            ''' update G '''
            for p in netD.parameters(): 
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            ''' Note that by setting this, we use different fake images in G and D updates '''
            #z.data.normal_(0, 1) # resample random noises
            #fake_images, mean_var = to_img_dict(data_parallel(netG, (embeddings, z)))

            loss_val  = 0
            kl_loss = get_KL_Loss(mean_var[0], mean_var[1])
            kl_loss_val = kl_loss.cpu().data.numpy().mean()
            generator_loss = args.KL_COE*kl_loss
            kl_loss_plot.plot(kl_loss_val)
            for key, _ in fake_images.items():
                # iterate over image of different sizes.
                this_fake  = fake_images[key]
                fake_pair_logit, fake_img_logit_global = data_parallel(netD, (this_fake,  embeddings))

                real_labels, _ = get_labels(fake_pair_logit)
                generator_loss += compute_g_loss(fake_pair_logit, real_labels)

                real_labels, _ = get_labels(fake_img_logit_global)
                img_loss = compute_g_loss(fake_img_logit_global, real_labels)

                generator_loss += img_loss * image_loss_ratio

            generator_loss.backward()
            optimizerG.step()
            netG.zero_grad()

            g_loss_val = generator_loss.cpu().data.numpy().mean()
            g_loss_plot.plot(g_loss_val)
            lr_plot.plot(g_lr)
            plot_dict['gen'].append(g_loss_val)
            gen_iterations += 1
            global_iter += 1
            sys.stdout.flush()
            # visualize train samples
            if it % args.verbose_per_iter == 0:
                for k, sample in fake_images.items():
                    # plot_imgs(sample.cpu().data.numpy(), epoch, k, 'train_samples')
                    plot_imgs([images[k].numpy(), sample.cpu().data.numpy()], epoch, k, 'train_images')
                print ('[epoch %d/%d iter %d/%d]: lr = %.6f g_loss = %.5f d_loss= %.5f kl_loss: %.5f' % (epoch, tot_epoch, it, updates_per_epoch, g_lr, g_loss_val, d_loss_val, kl_loss_val))
                sys.stdout.flush()

        ''' visualize test per epoch '''
        # generate samples
        gen_samples = []
        img_samples = []
        vis_samples = {}
        netG.eval()
        for idx_test in range(num_test_forward + 1):
            #sent_emb_test, _ =  netG.condEmbedding(test_embeddings)
            if idx_test == 0:
                test_images, test_embeddings = fixed_images, fixed_embeddings
            else:
                test_images, _, test_embeddings, _, _ = test_sampler.next()
                test_embeddings = to_device(test_embeddings) 
                testing_z = Variable(test_z.data, volatile=True)

            tmp_samples = {}

            for t in range(args.test_sample_num):

                if idx_test == 0: # plot fixed
                    testing_z = fixed_z_list[t]
                else:
                    testing_z.data.normal_(0, 1)

                fake_images, _ = to_img_dict(data_parallel(netG, (test_embeddings, testing_z)))
                samples = fake_images
                if idx_test == 0 and t == 0:
                    for k in samples.keys():
                        vis_samples[k] = [None for i in range(args.test_sample_num + 1)] # +1 to fill real image

                # Oops! very tricky to organize data for plot inputs!!!
                # vis_samples[k] = [real data, sample1, sample2, sample3, ... sample_args.test_sample_num]
                for k, v in samples.items():
                    cpu_data = v.cpu().data.numpy()
                    if t == 0:
                        if vis_samples[k][0] is None:
                            vis_samples[k][0] = test_images[k].numpy()
                        else:
                            vis_samples[k][0] =  np.concatenate([vis_samples[k][0], test_images[k].numpy() ], 0)

                    if vis_samples[k][t+1] is None:
                        vis_samples[k][t+1] = cpu_data
                    else:
                        vis_samples[k][t+1] = np.concatenate([vis_samples[k][t+1], cpu_data], 0)

        end_timer = time.time() - start_timer
        # visualize samples
        for typ, v in vis_samples.items():
            plot_imgs(v, epoch, typ, 'test_samples', path=model_folder)

        # save weights
        if epoch % args.save_freq == 0:
            netD = netD.cpu()
            netG = netG.cpu()
            torch.save(netD.state_dict(), os.path.join(model_folder, 'D_epoch{}.pth'.format(epoch)))
            torch.save(netG.state_dict(), os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch)))
            print('save weights at {}'.format(model_folder))
            torch.save(plot_dict, plot_save_path)
            netD = netD.cuda()
            netG = netG.cuda()
        print ('epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))
