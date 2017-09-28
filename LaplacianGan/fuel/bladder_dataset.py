import numpy as np
from copy import copy
from torch.autograd import Variable
import random, pickle
from ..proj_utils.local_utils import pre_process_img, process_sent, split_words

class GansData():
    def __init__(self, data, pickle_path, Index=None, batch_size=128, 
                 dictionary=None, random=True, loop=False):
        '''
        data: a hdf5 opened pointer
        Index: one array index, [3,5,1,9, 100] index of candidates 
        #Embedding: nn.Module that transfer words list (length of n) as emebdding mat (n, emb_dim)
        '''
        self.__dict__.update(locals())

        self.images   = self.data['images']
        self.sentence = self.data['sentences'].value

        with open(pickle_path, 'r') as f:
            self.sent_emb = pickle.load(f)   # B, nsent, dim
        
        self.emb_dim = self.sent_emb.shape[-1]

        self.eos_token = '<eos>'
        if self.Index is None:
            self.Index =  np.arange(0, self.images.shape[0])
        
        self.Totalnum   = len(self.Index)
        self.true_data  = np.zeros((batch_size,) + self.images.shape[1::], dtype=np.float32)
        self.wrong_data = np.zeros((batch_size,) + self.images.shape[1::], dtype=np.float32)
        self.sent_data  = np.zeros((batch_size, self.emb_dim ), dtype=np.float32)

        self.reset()

    def reset(self):
        self.chunkstart = 0
        sample_rand_Ind = copy(self.Index)
        if self.random:
            random.shuffle(sample_rand_Ind)
        self.totalIndx = sample_rand_Ind
        
        self.numberofchunk = (self.Totalnum + self.batch_size - 1) // self.batch_size  # the floor
        self.chunkidx = 0
    
    def __next__(self):

        if self.chunkidx >= self.numberofchunk:
            if self.loop:
                raise StopIteration
            else:
                self.reset()

        thisnum = min(self.batch_size, self.Totalnum - self.chunkidx * self.batch_size)
        curr_indices = self.totalIndx[self.chunkstart: self.chunkstart + thisnum]
        self.cur_ind = curr_indices
        self.chunkstart += thisnum
        self.chunkidx += 1
        seqs = []
        embeding_list = []

        for ind, idx in enumerate(curr_indices):
            self.true_data[ind]  = self.images[idx]
            rand_idx = self.totalIndx[random.randint(0, self.Totalnum-1)]
            self.wrong_data[ind] = self.images[rand_idx]
            
            this_emblist = self.sent_emb[idx]
            total_caps    =  this_emblist.shape[0]
            this_emb   = this_emblist[random.randint(0, total_caps-1)]
            self.sent_emb[idx] = this_emb

            this_sentlist = self.sentence[idx]
            this_sent = this_sentlist[random.randint(0, total_caps-1)]
            seqs.append(this_sent)

        images_dict = {}
        wrongs_dict = {}
        for size in [256]:
            tmp = resize_images(self.true_data[0:thisnum], shape=[size, size])
            tmp = tmp * (2. / 255) - 1.
            images_dict['output_{}'.format(size)] = tmp
            tmp = resize_images(self.wrong_data[0:thisnum], shape=[size, size])
            tmp = tmp * (2. / 255) - 1.
            wrongs_dict['output_{}'.format(size)] = tmp
        
        return sampled_images, sampled_embeddings_batchs, None, seqs

    def __iter__(self):
        return self

class CapData():
    def __init__(self, data, batch_size=128, split_dict=None,
                 refer_dict=None, dictionary=None, cuda=False):
        '''
        Construct train, valid and test iterator

        Parameters:
        -----------
        data: a hdf5 opened pointer
        split_dict:  dictionary{train:[list of file name], test:[], valid: []} 
        refer_dict:  list of dictionary of {filename:ind}, used to get_split_ind
        dictionary: {word:id}
        '''

        self.__dict__.update(locals())

        self.get_split_ind()

    def get_split_ind(self):
        '''
        transfer split file name list to list of index.
        '''
        self.train = []
        self.test = []
        self.valid = []
        self.all = []

        for name in self.split_dict.get('train', []):
            self.train.append(self.refer_dict[name])

        for name in self.split_dict.get('test', []):
            self.test.append(self.refer_dict[name])

        for name in self.split_dict.get('valid', []):
            self.valid.append(self.refer_dict[name])

        for name in self.split_dict.get('all', []):
            self.all.append(self.refer_dict[name])

    def get_flow(self, split='train'):
        if split == 'train':
            self.train_cls = GansData(self.data, self.train, self.batch_size)
            return self.train_cls

        if split == 'test':
            self.test_cls = GansData(self.data, self.test, self.batch_size)
            return self.test_cls

        if split == 'valid':
            self.valid_cls = GansData(self.data, self.valid, self.batch_size)
            return self.valid_cls

        if split == 'all':
            self.all_cls = GansData(self.data, self.all, self.batch_size)
            return self.all_cls