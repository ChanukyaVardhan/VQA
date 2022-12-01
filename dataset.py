from PIL import Image
from torch.utils.data import Dataset
from utils import pad_sequences

import collections
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as transforms

class VQADataset(Dataset):
    
    def __init__(self, data_dir, transform = None, mode = 'train', use_image_embedding = True, top_k = 1000, max_length = 14):
        """
            - data_dir:            directory of images and preprocessed data
            - transform:           any transformations to be applied to image (if not using embeddings)
            - mode:                train/val
            - use_image_embedding: use image embeddings directly that are stored
            - top_k:               select top_k frequent answers for training
            - max_length:          max number of words in the question to use while training
        """
        self.data_dir              = data_dir
        self.transform             = transform
        self.mode                  = mode
        self.use_image_embedding   = use_image_embedding
        
        self.labelfreq             = pickle.load(open(os.path.join(data_dir, f'answers_freqs.pkl'), 'rb'))
        self.label2idx             = {x[0]: i+1 for i, x in enumerate(collections.Counter(self.labelfreq).most_common(n = top_k - 1))}
        self.label2idx["<unk>"]    = 0

        self.word2idx              = pickle.load(open(os.path.join(data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"]
        self.max_length            = max_length
        
        self.data_file             = f'{mode}_data.txt'
        self.img_dir               = f'{mode}2014'

        # Read the processed data file
        with open(os.path.join(data_dir, self.data_file), 'r') as f:
            self.data              = f.read().strip().split('\n')

        self.image_features        = None
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id, question, answer, all_answers = self.data[idx].strip().split('\t')
        
        if not self.use_image_embedding: # If not use embedding, load the image and apply transform
            img = Image.open(f"{self.data_dir}/images/{self.img_dir}/COCO_{self.img_dir}_{int(image_id):012d}.jpg")
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        else: # if use embedding, directly load the embedding vector for VGG/ResNet
            if self.image_features == None:
                # NEED A WAY TO DISTINGUISH BETWEEN VGG AND RESNET, PROBABLY KEEP THESE IN A FOLDER
                self.image_features = pickle.load(open(os.path.join(self.data_dir, f'{self.mode}_image_embeddings_new.pkl'), 'rb'))
            img  = self.image_features[image_id]
        
        # convert question words to indexes
        question = [self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] for w in question.split()]
        question = pad_sequences(question, self.max_length)

        # convert answer words to indexes
        answer   = self.label2idx[answer if answer in self.label2idx else '<unk>']

        # convert all 10 answers to tokens
        all_answers = all_answers.strip().split("^")
        all_answers = np.array([self.label2idx[a if a in self.label2idx else '<unk>'] for a in all_answers])

        # calculate individual answer's frequency and calculate soft score for them
        ans_freqs   = {}
        for a in all_answers:
            ans_freqs[a] = ans_freqs.get(a, 0) + 1
        # soft_score  = [(a, min(1, freq / 3)) for a, freq in ans_freqs.items() if a != 0] # Ignore unknowns
        soft_score  = [(a, min(1, freq / 3)) for a, freq in ans_freqs.items()]
        ans_score   = np.zeros(len(self.label2idx), dtype=np.float32)
        for a, s in soft_score:
            ans_score[a] = s

        return img, question, answer, all_answers, ans_score
