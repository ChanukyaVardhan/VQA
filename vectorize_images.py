from __future__ import print_function
import errno
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import collections
import numpy as np
import pickle
from torch.autograd import Variable
import time

def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def vectorize(dataset, image_ids, normalize=False, output_size=1024):
    img_id_embedding_map = {}
    # Load the pretrained model
    model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(*list(model.classifier)[:-1])
    transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
    i = 0
    for img_id in image_ids:
        image_path = data_dir + dataset + '/COCO_' + dataset + "_" + "0"*(12-len(img_id)) + img_id + '.jpg'
        img = pil_loader(image_path)
        
        img_tensor = Variable(transform(img).unsqueeze(0))
        i+=1
        print(i)
        embedding = model(img_tensor)
        embedding = embedding.detach().numpy()[0]
        img_id_embedding_map[img_id] = embedding
    
    print("saving image embeddings file for ",dataset," ... ")
    with open(os.path.join(data_dir, dataset + "_image_embeddings.pkl"), 'wb') as f:
        pickle.dump(img_id_embedding_map, f)

def get_ids(dataset):
    f = open(dataset)
    data = f.read().split('\n')
    image_ids = [row.split('\t')[0] for row in data]
    image_ids = image_ids[:len(image_ids)-1]     
    return image_ids

if __name__ == "__main__":
    data_dir         = "images/"
    train_image_id_list = get_ids('train_data.txt')    
    test_image_id_list = get_ids('test_data.txt')    
    val_image_id_list = get_ids('val_data.txt')    
    vectorize('train2014', train_image_id_list)
    vectorize('test2015', test_image_id_list)
    vectorize('val2014', val_image_id_list)