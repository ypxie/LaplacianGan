import glob
import os
import torchfile as torchf
import cPickle

path = 'Data/coco/'
save_path = path + 'train/'

t7list = glob.glob(os.path.join(path, 'val2014_ex_t7/*.t7'))
# {'char': not used , 'txt': embedding, 'img': name of file }

embeding_fn = 'char-CNN-RNN-embeddings.pickle' # list([numofembedding, 1024])
file_info = 'filenames.pickle' # list('filename.jpg')

# image need to save in folder not 

File_info = []
Embed = []
k = 0
for f in t7list:
    data = torchf.load(f)   
    name = data['img']
    embeddings = data['txt']
    Embed.append(embeddings.copy())
    File_info.append(name)
    k += 1
    if k % 100 == 0:
        print('append {}'.format(k),  name)

cPickle.dump(File_info, open(os.path.join(save_path, file_info), 'wb'))
cPickle.dump(Embed, open(os.path.join(save_path, embeding_fn),  'wb'))

