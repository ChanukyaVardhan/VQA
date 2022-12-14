import optuna
from dataset import VQADataset
from models.baseline import VQABaseline
from PIL import Image
from torch.optim import Adam, lr_scheduler, RMSprop, SGD
from torch.utils.data import Dataset, DataLoader
from train import *
from utils import *

import argparse
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms

def get_model(model_type, vocab_size, use_image_embedding):
    """
        Instantiates the pytorch model given the appropriate parameters.
    """
    model = None

    if model_type == 'baseline':
        model = VQABaseline(vocab_size = vocab_size, use_image_embedding = use_image_embedding)
    else:
        raise Exception(f'Model Type {model_type} is not supported')

    return model

def objective(trial):
    """
    specifiy a trial - hyperparameters to tune, list of choices we want to explore for each
    build the model and return accuracy as a metric so that accuracy can be maximized across each trial
    """
    params = {
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              }
    
    model = build_model(params)

    accuracy = train_and_evaluate(params, model)

    return accuracy

def build_model(params):
    """
    build the VQA model and return 
    """
    model        = get_model(args.model, vocab_size, args.use_image_embedding)

    return model

def train_and_evaluate(params, model):
    """
    run a grid search to find the best set of hyper-parameters
    """
    model        = nn.DataParallel(model).to(device) if num_gpus > 1 else model.to(device)
    optimizer    = params['optimizer']
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr = args.learning_rate)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr = args.learning_rate)
    elif optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr = args.learning_rate)
    loss_fn      = nn.CrossEntropyLoss()
    # FIX THE SCHEDULER
    scheduler    = lr_scheduler.MultiStepLR(optimizer, milestones = [60, 120, 180], gamma = 0.1)

    model, optim, best_accuracy = \
        train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device,
                    args.model_dir, args.log_dir, epochs = args.epochs,
                    run_name = args.run_name, save_best_state = args.save_best_state,
                    print_stats = args.print_stats, print_epoch_freq = args.print_epoch_freq,
                    print_step_freq = args.print_step_freq)

    parse_tb_logs(args.log_dir, args.run_name, 'epoch')
    parse_tb_logs(args.log_dir, args.run_name, 'step')
    return best_accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VQA')
    parser.add_argument('--data_dir',             type=str,   help='directory of preprocesses data', default='/scratch/an3729/datasets')
    parser.add_argument('--model_dir',            type=str,   help='directory to store model checkpoints', default='/scratch/an3729/checkpoints')
    parser.add_argument('--log_dir',              type=str,   help='directory to store log files', default='/scratch/an3729/logs')
    parser.add_argument('--run_name',             type=str,   help='unique experiment setting name', default='grid_search', required=True)
    parser.add_argument('--model',                type=str,   help='VQA model choice', choices=['baseline'], default='baseline', required=True)

    parser.add_argument('--use_image_embedding',  type=bool,  help='use pre computed image embeddings', default=True)
    parser.add_argument('--top_k_answers',        type=int,   help='top k answers', default=1000)
    parser.add_argument('--max_length',           type=int,   help='max sequence length of questions', default=14) # covers 99.7% of questions

    parser.add_argument('--batch_size',           type=int,   help='batch size per CPU/GPU', default=64)
    parser.add_argument('--epochs',               type=int,   help='number of epochs i.e., final epoch number', default=50)
    parser.add_argument('--beta1',                type=float, help='beta1 parameter', default=0.9)
    parser.add_argument('--beta2',                type=float, help='beta2 parameter', default=0.999)
    parser.add_argument('--learning_rate',        type=float, help='initial learning rate', default=0.001)

    parser.add_argument('--print_stats',          type=bool,  help='flag to print statistics', default=True)
    parser.add_argument('--print_epoch_freq',     type=int,   help='epoch frequency to print stats', default=1)
    parser.add_argument('--print_step_freq',      type=int,   help='step frequency to print stats', default=300)
    parser.add_argument('--save_best_state',      type=bool,  help='flag to save best model', default=False)

    parser.add_argument('--random_seed',          type=int,   help='random seed', default=43)

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform    = transforms.Compose([
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_ds     = VQADataset(args.data_dir, top_k = args.top_k_answers, max_length = args.max_length, transform = transform, use_image_embedding = args.use_image_embedding)
    val_ds       = VQADataset(args.data_dir, mode = 'val', top_k = args.top_k_answers, max_length = args.max_length, transform = transform, use_image_embedding = args.use_image_embedding)

    num_gpus     = torch.cuda.device_count()
    batch_size   = args.batch_size
    if num_gpus: # Same batch size on each GPU ,i.e, weak scaling
        batch_size *= num_gpus
    train_loader = None
    val_loader   = None
    # create dataset loaders
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
    val_loader   = DataLoader(val_ds, batch_size = batch_size, num_workers = 2, pin_memory = True)
    vocab_size   = len(pickle.load(open(os.path.join(args.data_dir, 'questions_vocab.pkl'), 'rb'))["word2idx"])
    # create an optuna study, that will maximize accuracy by runnning various trials with the hyper-parameter choices given
    # the runs will be terminated if the accuracy is not improving for certain consecutive epochs
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=10)
    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))


# python grid_search.py --run_name testrun --model baseline --data_dir ../Dataset --model_dir ../checkpoints --log_dir ../logs --epochs 1 