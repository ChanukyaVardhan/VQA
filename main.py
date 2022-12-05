from dataset import VQADataset
from models.baseline import VQABaseline
from PIL import Image
from torch.optim import Adadelta, Adam, lr_scheduler, RMSprop
from torch.utils.data import Dataset, DataLoader
from train import *
from utils import parse_tb_logs

import argparse
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms

def get_model(model_type, vocab_size, use_image_embedding, use_dropout, output_size, image_model_type, attention_mechanism, word_embedding_size, lstm_state_size):
    model = None

    if model_type == 'baseline':
        model = VQABaseline(vocab_size = vocab_size, use_image_embedding = use_image_embedding, use_dropout = use_dropout,
                            output_size = output_size, image_model_type = image_model_type, attention_mechanism = attention_mechanism,
                            word_embedding_size = word_embedding_size, lstm_hidden_size = lstm_state_size)
    else:
        raise Exception(f'Model Type {model_type} is not supported')

    return model

def boolstr(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser(description='VQA')

    parser.add_argument('--data_dir',               type=str,   help='directory of preprocesses data', default='/scratch/crg9968/datasets')
    parser.add_argument('--model_dir',              type=str,   help='directory to store model checkpoints', default='/scratch/crg9968/checkpoints')
    parser.add_argument('--log_dir',                type=str,   help='directory to store log files', default='/scratch/crg9968/logs')
    parser.add_argument('--run_name',               type=str,   help='unique experiment setting name', default='testrun', required=True)
    parser.add_argument('--model',                  type=str,   help='VQA model choice', choices=['baseline'], default='baseline', required=True)
    parser.add_argument('--image_model_type',       type=str,   help='Type of CNN', choices=['vgg16', 'resnet152'], default='vgg16')

    parser.add_argument('--use_image_embedding',    type=boolstr,  help='use pre computed image embeddings', default=True)
    parser.add_argument('--top_k_answers',          type=int,   help='top k answers', default=1000)
    parser.add_argument('--max_length',             type=int,   help='max sequence length of questions', default=14) # covers 99.7% of questions
    parser.add_argument('--word_embedding_size',    type=int,   help='Word embedding size', default=300)
    parser.add_argument('--lstm_state_size',        type=int,   help='LSTM hidden state size', default=512)

    parser.add_argument('--batch_size',             type=int,   help='batch size', default=512)
    parser.add_argument('--epochs',                 type=int,   help='number of epochs i.e., final epoch number', default=50)
    parser.add_argument('--learning_rate',          type=float, help='initial learning rate', default=1.0)
    parser.add_argument('--optimizer',              type=str,   help='choice of optimizer', choices=['adam', 'adadelta', 'rmsprop'], default='adadelta')
    parser.add_argument('--use_dropout',            type=boolstr,  help='use dropout', default=False)
    # parser.add_argument('--dropout_prob',         type=float, help='dropout probability', default=0.5)
    parser.add_argument('--use_sigmoid',            type=boolstr,  help='use sigmoid activation to compute binary cross entropy loss', default=False)
    parser.add_argument('--use_sftmx_multiple_ans', type=boolstr,  help='use softmax activation with multiple possible answers to compute the loss', default=False)
    parser.add_argument('--ignore_unknowns',        type=boolstr,  help='Ignore unknowns from the true labels in case of use_sigmoid', default=True)
    parser.add_argument('--use_softscore',          type=boolstr,  help='use soft score for the ansewrs, only applicable for sigmoid or softmax with multiple answers case', default=True)

    parser.add_argument('--print_stats',            type=boolstr,  help='flag to print statistics', default=True)
    parser.add_argument('--print_epoch_freq',       type=int,   help='epoch frequency to print stats', default=1)
    parser.add_argument('--print_step_freq',        type=int,   help='step frequency to print stats', default=300)
    parser.add_argument('--save_best_state',        type=boolstr,  help='flag to save best model', default=True)
    parser.add_argument('--attention_mechanism',    type=str,   help='method of combining image and text embeddings', default='element_wise_product')
    
    parser.add_argument('--random_seed',            type=int,   help='random seed', default=43)

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform    = transforms.Compose([
                       transforms.Resize((224, 224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.model == 'baseline':
        train_ds     = VQADataset(args.data_dir, top_k = args.top_k_answers, max_length = args.max_length, transform = transform,
                                  use_image_embedding = args.use_image_embedding, image_model_type = args.image_model_type,
                                  ignore_unknowns = args.ignore_unknowns, use_softscore = args.use_softscore)
        val_ds       = VQADataset(args.data_dir, mode = 'val', top_k = args.top_k_answers, max_length = args.max_length, transform = transform,
                                  use_image_embedding = args.use_image_embedding, image_model_type = args.image_model_type,
                                  ignore_unknowns = args.ignore_unknowns, use_softscore = args.use_softscore)
    else:
        raise Exception(f'Model Type {args.model} is not supported')

    num_gpus     = torch.cuda.device_count()
    batch_size   = args.batch_size
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
    val_loader   = DataLoader(val_ds, batch_size = batch_size, num_workers = 2, pin_memory = True)

    vocab_size   = len(pickle.load(open(os.path.join(args.data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"])
    model        = get_model(args.model, vocab_size, args.use_image_embedding, args.use_dropout, args.top_k_answers, args.image_model_type, args.attention_mechanism, args.word_embedding_size, args.lstm_state_size)
    model        = nn.DataParallel(model).to(device) if num_gpus > 1 else model.to(device)
    
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr = args.learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr = args.learning_rate)
    else:
        optimizer = Adadelta(model.parameters(), lr = args.learning_rate)

    if args.use_sigmoid:
        loss_fn   = nn.BCEWithLogitsLoss()
    elif args.use_sftmx_multiple_ans:
        loss_fn   = nn.LogSoftmax()
    else:
        loss_fn   =  nn.CrossEntropyLoss()

    model, optim, best_accuracy = \
        train_model(model, train_loader, val_loader, loss_fn, optimizer, device,
                    args.model_dir, args.log_dir, args.learning_rate, epochs = args.epochs,
                    run_name = args.run_name, use_sigmoid = args.use_sigmoid, use_sftmx_multiple_ans = args.use_sftmx_multiple_ans,
                    save_best_state = args.save_best_state, print_stats = args.print_stats,
                    print_epoch_freq = args.print_epoch_freq, print_step_freq = args.print_step_freq)

    model, vqa_accuracy = get_VQA_accuracy(model, val_loader, device)

    parse_tb_logs(args.log_dir, args.run_name, 'epoch')
    parse_tb_logs(args.log_dir, args.run_name, 'step')

if __name__ == '__main__':
    main()

# python3 main.py --run_name testrun --model baseline --data_dir ../Dataset --model_dir ../checkpoints --log_dir ../logs --epochs 4
