import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter

from torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
from .proj_utils.local_utils import *

from .proj_utils.torch_utils import *
from .zzGan import load_partial_state_dict

from PIL import Image, ImageDraw, ImageFont

import time, json, h5py

TINY = 1e-8

def drawCaption(img, caption, level=['output 64', 'output 128', 'output 256']):
    img_txt = Image.fromarray(img)
    # get a font
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)

    # draw text, half opacity
    for idx, this_level in enumerate(level):
        d.text((10, 256 + idx * 256), this_level, font=fnt, fill=(255, 255, 255, 255))

    #d.text((10, 256), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
    #d.text((10, 512), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))
    #if img.shape[0] > 832:
    #    d.text((10, 832), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
    #    d.text((10, 1088), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))

    idx = caption.find(' ', 60)
    if idx == -1:
        d.text((256, 10), caption, font=fnt, fill=(255, 255, 255, 255))
    else:
        cap1 = caption[:idx]
        cap2 = caption[idx+1:]
        d.text((256, 10), cap1, font=fnt, fill=(255, 255, 255, 255))
        d.text((256, 60), cap2, font=fnt, fill=(255, 255, 255, 255))

    return img_txt

def save_super_images(vis_samples, captions_batch, batch_size, save_folder, saveIDs):
    dst_shape = (0,0)
    all_row = []
    level = []
    for typ, img_list in vis_samples.items():
        this_shape = img_list[0].shape[2::] # bs, 3, row, col
        if this_shape[0] > dst_shape[0]:
            dst_shape = this_shape
        level.append(typ)

    valid_caption = []
    valid_IDS = []
    for j in range(batch_size):
        if not re.search('[a-zA-Z]+', captions_batch[j]):
            continue
        else:  
            valid_caption.append(captions_batch[j])
            valid_IDS.append(saveIDs[j])
    
    for typ, img_list in vis_samples.items(): 
        img_tensor = np.stack(img_list, 1) # N * T * 3 *row*col
        img_tensor = img_tensor.transpose(0,1,3,4,2)
        img_tensor = (img_tensor + 1.0) * 127.5
        img_tensor = img_tensor.astype(np.uint8)

        batch_size  = img_tensor.shape[0]
        #imshow(img_tensor[0,0])
        batch_all = []
        for bidx in range(batch_size):  
            if not re.search('[a-zA-Z]+', captions_batch[j]):
                continue
            padding = np.zeros(dst_shape + (3,), dtype=np.uint8)
            this_row = [padding]
            # First row with up to 8 samples
            for tidx in range(np.minimum(8, img_tensor.shape[1] )):
                this_img  = img_tensor[bidx][tidx]
                re_sample = imresize_shape(this_img, dst_shape)
                this_row.append(re_sample)
                
            this_row = np.concatenate(this_row, axis=1) # row, col*T, 3
            batch_all.append(this_row)
        batch_all = np.stack(batch_all, 0) # bs*row*colT*3 
        all_row.append(batch_all)

    all_row = np.stack(all_row, 0) # n_type * bs * shape    
    

    batch_size = len(valid_IDS)

    for idx in range(batch_size):
        this_select = all_row[:, idx] # ntype*row*col
        
        ntype, row, col, chn = this_select.shape
        superimage = np.reshape(this_select, (-1, col, chn) )  # big_row, col, 3

        top_padding = np.zeros((128, superimage.shape[1], 3))
        superimage =\
            np.concatenate([top_padding, superimage], axis=0)
            
        save_path = os.path.join(save_folder, '{}.png'.format(valid_IDS[idx]) )    
        superimage = drawCaption(np.uint8(superimage), valid_caption[idx], level)
        scipy.misc.imsave(save_path, superimage)



def generate_layer_features(dataset, model_root, mode_name, save_root , netG,  args):

    print('Using testing mode')
    netG.eval()

    test_sampler  = dataset.test.next_batch_test

    model_folder = os.path.join(model_root, mode_name)
    model_marker = mode_name + '_G_epoch_{}'.format(args.load_from_epoch)
    
    ''' load model '''
    assert args.load_from_epoch != '', 'args.load_from_epoch is empty'
    G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
    print('reload weights from {}'.format(G_weightspath))
    weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
    load_partial_state_dict(netG, weights_dict)
    
    testing_z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    testing_z = to_device(testing_z, netG.device_id, volatile=True)


    num_examples = 100
    start_count = 0
    while True:
        if start_count >= num_examples:
            break
        test_images, test_embeddings_list, saveIDs, test_captions = test_sampler(args.batch_size, start_count, 1)
        imname = str(saveIDs[0])
        this_batch_size = test_images.shape[0]
        # print('start: {}, this_batch size {}, num_examples {}'.format(start_count, test_images.shape[0], dataset.test._nu   m_examples  ))
       
        start_count += this_batch_size
        
        #test_embeddings_list is a list of (B,emb_dim)
        ''' visualize test per epoch '''
        # generate samples
        
        for t in range(1):
            testing_z.data.normal_(0, 1)

            this_test_embeddings = test_embeddings_list[0] # just use the first one
            this_test_embeddings = to_device(this_test_embeddings, netG.device_id, volatile=True)
            test_outputs, _ = netG(this_test_embeddings, testing_z[0:this_batch_size])
            images = []
            # import pdb; pdb.set_trace()

            for k, v in test_outputs.items():
                dim = k.split('_')[1]
                feat = getattr(netG, 'keep_out_'+dim).cpu().data[0]
                import pdb; pdb.set_trace()
                img = (v.cpu().data[0].numpy().transpose((1,2,0)) + 1) / 2
                images.append(misc.imresize(img, (256,256)))
                num = feat.size()[0]
                
                res = make_grid(feat.unsqueeze(1), nrow=int(np.sqrt(num)))
                res = res.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

                misc.imsave(os.path.join(save_root, imname+'_feat_'+dim+'.png'), res)
            import pdb; pdb.set_trace()
            images = np.concatenate(images, axis=1)
            misc.imsave(os.path.join(save_root, imname+'_images.png'), images)



                


                

def test_gans(dataset, model_root, mode_name, save_root , netG,  args):
    # helper function
    if args.train_mode:
        print('Using training mode')
        netG.train()
    else:
        print('Using testing mode')
        netG.eval()

    test_sampler  = dataset.test.next_batch_test

    model_folder = os.path.join(model_root, mode_name)
    model_marker = mode_name + '_G_epoch_{}'.format(args.load_from_epoch)

    save_folder  = os.path.join(save_root, model_marker )   # to be defined in the later part
    save_h5    = os.path.join(save_root, model_marker+'.h5')
    org_h5path = os.path.join(save_root, 'original.h5')
    mkdirs(save_folder)
    
    ''' load model '''
    assert args.load_from_epoch != '', 'args.load_from_epoch is empty'
    G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
    print('reload weights from {}'.format(G_weightspath))
    weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
    load_partial_state_dict(netG, weights_dict)
    
    testing_z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    testing_z = to_device(testing_z, netG.device_id, volatile=True)

    num_examples = dataset.test._num_examples
    dataset.test._num_examples = num_examples
    
    total_number = num_examples * args.test_sample_num
    init_flag = True

    all_choosen_caption = []
    file_not_exists = not os.path.exists(org_h5path)

    if file_not_exists:
        org_h5 = h5py.File(org_h5path,'w')
        org_dset = org_h5.create_dataset('output_256', shape=(num_examples,256, 256,3), dtype=np.uint8)
    else:
        org_dset = None
    with h5py.File(save_h5,'w') as h5file:
        
        start_count = 0
        data_count = {}
        dset = {}

        
        while True:
            if start_count >= num_examples:
                break
            test_images, test_embeddings_list, saveIDs, test_captions = test_sampler(args.batch_size, start_count, 1)
            
            this_batch_size =  test_images.shape[0]
            #print('start: {}, this_batch size {}, num_examples {}'.format(start_count, test_images.shape[0], dataset.test._num_examples  ))
            chosen_captions = []
            for this_caption_list in test_captions:
                chosen_captions.append(this_caption_list[0])

            all_choosen_caption.extend(chosen_captions)    
            if org_dset is not None:
                org_dset[start_count:start_count+this_batch_size] = ((test_images + 1) * 127.5 ).astype(np.uint8)
            
            start_count += this_batch_size
            
            #test_embeddings_list is a list of (B,emb_dim)
            ''' visualize test per epoch '''
            # generate samples
            gen_samples = []
            img_samples = []
            vis_samples = {}
            tmp_samples = {}
            
            for t in range(args.test_sample_num):
                
                B = len(test_embeddings_list)
                ridx = random.randint(0, B-1)
                testing_z.data.normal_(0, 1)

                this_test_embeddings = test_embeddings_list[ridx]
                this_test_embeddings = to_device(this_test_embeddings, netG.device_id, volatile=True)
                test_outputs, _ = netG(this_test_embeddings, testing_z[0:this_batch_size])
                
                if  t == 0:  
                    for k in test_outputs.keys():
                        vis_samples[k] = [None for i in range(args.test_sample_num)] # +1 to fill real image
                        img_shape = test_outputs[k].size()[2::]
                        
                        if init_flag is True:
                            #if '256' in k:
                            print('total number of images is: ', total_number)
                            dset[k] = h5file.create_dataset(k, shape=(total_number,)+ img_shape + (3,), dtype=np.uint8)
                            data_count[k] = 0
                            

                    init_flag = False    

                for k, v in test_outputs.items():
                    cpu_data = v.cpu().data.numpy()
                    # if t == 0:
                    #     if vis_samples[k][0] == None:
                    #         vis_samples[k][0] = test_images
                    #     else:
                    #         vis_samples[k][0] =  np.concatenate([ vis_samples[k][0], test_images], 0) 
                    
                    if vis_samples[k][t] == None:
                        vis_samples[k][t] = cpu_data
                    else:
                        vis_samples[k][t] = np.concatenate([vis_samples[k][t+1], cpu_data], 0)
            
            #save_super_images(vis_samples, chosen_captions, this_batch_size, save_folder, saveIDs)

            for typ, img_list in vis_samples.items():
                #print('img list lenght is: ', len(img_list))
                #img_tensor = np.stack(img_list, 1) # N * T * 3
                #data_dict[typ].append( np.stack(v, 1)  ) # list of N*T*3*row*col
                for this_img in img_list:
                    bs = this_img.shape[0]
                    
                    start = data_count[typ]
                    this_sample = ((this_img + 1) * 127.5 ).astype(np.uint8)
                    
                    this_sample = this_sample.transpose(0, 2,3,1)
                    #print(start, start+bs, this_sample.shape)

                    dset[typ][start: start + bs] = this_sample

                    data_count[typ] = start + bs
                    
            print('saved files: ', data_count)  
            
        caption_array = np.array(all_choosen_caption, dtype=object)
        string_dt = h5py.special_dtype(vlen=str)
        h5file.create_dataset("captions", data=caption_array, dtype=string_dt)
        if org_dset is not None:
            org_h5.close() 


#  def massive_samples(self, sess, dataset, save_dir, subset='test'):
#         import deepdish as dd
#         count = 0
#         print('num_examples:', dataset._num_examples)
#         print ('# of copies per sample: ', cfg.TRAIN.NUM_COPY)
#         print ('')
#         save_dir = './massive_samples'
#         save_path = os.path.join(save_dir, 'test_large_samples_{}'.format(cfg.TRAIN.NUM_COPY*dataset._num_examples))
#         if not os.path.isdir(save_path):
#             print ('creat folder ',save_path)
#             os.makedirs(save_path)

#         current_label = 1
#         TENSOR = np.zeros((5000, 256, 256, 3 ), np.uint8)
#         TENSOR_C = 0
#         TOT = 0
#         assert(self.batch_size == 1)
#         while count < dataset._num_examples:
#             start = count % dataset._num_examples
#             images, embeddings_batchs, savenames, captions_batchs =\
#                 dataset.next_batch_test(self.batch_size, start, 1)
            
#             # the i-th sentence/caption
#             label = int(savenames[0].split('.')[0])
#             name = os.path.basename(savenames[0])
#             # cls_save_path = os.path.join(save_path, str(label))
#             # if not os.path.isdir(cls_save_path):
#             #     print ('creat folder ', cls_save_path)
#             #     os.mkdir(cls_save_path)

#             count += self.batch_size 

#             # print ('current label', label)
#             # if label != 197: continue
#             # current_label = label

#             if label != current_label:
#                 print ('move from label {} to {}'.format(current_label, label))
#                 sp = os.path.join(save_path, 'label_{}_{}.h5'.format(TENSOR_C, current_label))
#                 dd.io.save(sp, {'samples': TENSOR[:TENSOR_C]})
#                 print ('save tensor to ', sp)
#                 TENSOR.fill(0)
#                 TENSOR_C = 0
#                 current_label = label

#             samples_batchs = []
#             hr_samples_batchs = []
#             # Generate up to 16 images for each sentence,
#             # with randomness from noise z and conditioning augmentation.
#             numSamples = np.minimum(16, cfg.TRAIN.NUM_COPY)  

#             for j in range(numSamples):
#                 hr_samples, samples =\
#                     sess.run([self.hr_fake_images, self.fake_images],
#                                 {self.embeddings: embeddings_batchs})

#                 hr_samples = (hr_samples[0] + 1) * 127.5
#                 hr_samples_batchs.append(hr_samples)
#                 TENSOR[TENSOR_C] = hr_samples.astype(np.uint8)
#                 TENSOR_C += 1
#                 TOT += 1
            
#                 # scipy.misc.imsave(os.path.join(cls_save_path, '{}_copy{}.jpg'.format(name, j)), hr_samples)
#             print('label {}, TENSOR_C {}'.format(label, TENSOR_C))

#         # save the last one     
#         sp = os.path.join(save_path, 'label_{}_{}.h5'.format(TENSOR_C, label))
#         dd.io.save(sp, {'samples': TENSOR[:TENSOR_C]})
#         print ('save tensor to ', sp)
#         print ('{} sampeles are generated'.format(TOT))​
