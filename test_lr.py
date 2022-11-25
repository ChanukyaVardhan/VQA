from dataset import VQADataset
from models.baseline import VQABaseline
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import *

import argparse
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

def lr_explore(model, train_loader, loss_fn, device, model_dir, log_dir,
               epochs = 5, run_name = 'lr_adam'):
    lr_searches   = []
    loss_searches = []
    avg_loss      = 0
    beta          = 0.98
    start_lr      = 1e-9
    end_lr        = 1e1
    train_steps   = np.ceil(len(train_loader.dataset) / float(train_loader.batch_size))
    num_steps     = epochs * train_steps
    lr_multiplier = (end_lr / start_lr) ** (1.0 / num_steps)

    optimizer     = Adam(model.parameters(), lr = 1e-9)

    for epoch in range(1, epochs + 1):
        completed_steps = (epoch - 1) * train_steps

        for step, (images, questions, answers, all_answers) in enumerate(train_loader):
            images           = images.to(device)
            questions        = questions.to(device)
            answers          = answers.to(device)

            optimizer.zero_grad()

            pred_scores      = model(images, questions)
            l                = loss_fn(pred_scores, answers)

            l.backward()
            optimizer.step()
        
            completed_steps += 1

            optim_lr         = optimizer.param_groups[0]['lr']
            lr_searches.append(optim_lr)
            
            avg_loss         = beta * avg_loss + (1 - beta) * l.item()
            smooth_loss      = avg_loss / (1 - beta**completed_steps)
            loss_searches.append(smooth_loss)
            
            optim_lr        *= lr_multiplier
            for param_group in optimizer.param_groups:
                param_group['lr'] = optim_lr

    return lr_searches, loss_searches

def main():
    parser = argparse.ArgumentParser(description='Test learning rate for adam')

    parser.add_argument('--data_dir',             type=str,   help='directory of preprocesses data', default='/scratch/crg9968/datasets')
    parser.add_argument('--model_dir',            type=str,   help='directory to store model checkpoints', default='/scratch/crg9968/checkpoints')
    parser.add_argument('--log_dir',              type=str,   help='directory to store log files', default='/scratch/crg9968/logs')
    parser.add_argument('--run_name',             type=str,   help='unique experiment setting name', default='lr_adam')

    parser.add_argument('--top_k_answers',        type=int,   help='top k answers', default=1000)
    parser.add_argument('--max_length',           type=int,   help='max sequence length of questions', default=14) # covers 99.7% of questions

    parser.add_argument('--batch_size',           type=int,   help='batch size', default=512)
    parser.add_argument('--epochs',               type=int,   help='number of epochs i.e., final epoch number', default=5)

    parser.add_argument('--random_seed',          type=int,   help='random seed', default=43)

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds     = VQADataset(args.data_dir, top_k = args.top_k_answers, max_length = args.max_length)
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True, num_workers = 2, pin_memory = True)

    num_gpus     = torch.cuda.device_count()
    vocab_size   = len(pickle.load(open(os.path.join(args.data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"])
    model        = VQABaseline(vocab_size = vocab_size, use_image_embedding = True, use_dropout = False)
    model        = nn.DataParallel(model).to(device) if num_gpus > 1 else model.to(device)
    loss_fn      = nn.CrossEntropyLoss()

    lr_searches, loss_searches = \
            lr_explore(model, train_loader, loss_fn, device,
                       args.model_dir, args.log_dir,
                       epochs = args.epochs, run_name = args.run_name)

    df = pd.DataFrame({'lr': lr_searches, 'loss': loss_searches})
    df.to_csv(os.path.join(args.log_dir, args.run_name + '.csv'), index = False)

    plot_lr_explore_adam(args.log_dir, args.run_name)

if __name__ == '__main__':
    main()

# python3 test_lr.py --data_dir ../Dataset --model_dir ../checkpoints --log_dir ../logs
