from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import torch
import torchvision.transforms as transforms

def pad_sequences(l, max_length):
    padded = np.zeros((max_length,), np.int64)
    if len(l) > max_length:
        padded[:] = l[:max_length]
    else:
        padded[:len(l)] = l
    return padded

class VQADataset(Dataset):
    
    def __init__(self, data_dir, transform = None, mode = 'train', top_k = 1000):
        self.data_dir      		= data_dir
        self.transform     		= transform
        self.mode          		= mode
        
        self.labelfreq     		= pickle.load(open(os.path.join(data_dir, f'answers_freqs.pkl'), 'rb'))
        self.label2idx     		= {x[0]: i+1 for i, x in enumerate(collections.Counter(self.labelfreq).most_common(n = top_k))}
        self.label2idx["<unk>"] = 0

        self.word2idx      		= pickle.load(open(os.path.join(data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"]
        self.max_length    		= pickle.load(open(os.path.join(data_dir, 'questions_vocab.pkl'), 'rb'))["max_length"]
        
        self.data_file     		= None
        self.img_dir       		= None
        if mode == 'train':
            self.data_file 		= 'train_data.txt'
            self.img_dir   		= 'train2014'
        elif mode == 'val':
            self.data_file 		= 'val_data.txt'
            self.img_dir   		= 'val2014'
        else:
            self.data_file 		= None
            self.img_dir   		= None
            
        
        with open(self.data_file, 'r') as f:
            self.data 			= f.read().strip().split('\n')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id, question, answer = self.data[idx].strip().split('\t')
        
        img = Image.open(f"{self.data_dir}/images/{self.img_dir}/COCO_{self.img_dir}_{int(image_id):012d}.jpg")
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        question = [self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] for w in question.split()]
        question = pad_sequences(question, self.max_length)

        answer = self.label2idx[answer if answer in self.label2idx else '<unk>']

    	return img, question, answer
