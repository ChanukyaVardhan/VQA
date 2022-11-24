"""
Utility functions:
-> parse tensorboard event logs and save into csvs
-> generate statistics of length of questions
"""
from tensorboard.backend.event_processing import event_accumulator

import collections
import matplotlib.pyplot as plt
import os
import pandas as pd

def parse_tb_logs(log_directory, run_name, epoch_or_step = 'epoch'):
    """
        Given log directory and the expirement run name, gather the tensorboard
        events using event accumulator, extract the train and val statistics at
        either epoch or step level and saves them into a csv in the same log directory
    """
    if epoch_or_step == 'step':
        run_name   += '_step'

    train_losses      = []
    train_accuracies  = []
    val_losses        = []
    val_accuracies    = []

    directory         = os.path.join(log_directory, run_name)
    for filename in os.listdir(directory):
        ea = event_accumulator.EventAccumulator(os.path.join(directory, filename),
                                                size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        train_losses     += ea.Scalars('Train_Loss')
        train_accuracies += ea.Scalars('Train_Accuracy')
        if epoch_or_step == 'epoch':
            val_losses       += ea.Scalars('Val_Loss')
            val_accuracies   += ea.Scalars('Val_Accuracy')

    train_losses          = pd.DataFrame(train_losses).rename(columns = {'value': 'train_loss'})
    train_accuracies      = pd.DataFrame(train_accuracies).rename(columns = {'value': 'train_accuracy'})
    if epoch_or_step == 'epoch':
        val_losses            = pd.DataFrame(val_losses).rename(columns = {'value': 'val_loss'})
        val_accuracies        = pd.DataFrame(val_accuracies).rename(columns = {'value': 'val_accuracy'})

    df                    = train_losses
    df['train_accuracy']  = train_accuracies['train_accuracy']
    if epoch_or_step == 'epoch':
        df['val_loss']        = val_losses['val_loss']
        df['val_accuracy']    = val_accuracies['val_accuracy']
        df                    = df.rename(columns = {'step': 'epoch'})

    df = df.sort_values(by = 'wall_time', ascending = True)
    df = df.drop_duplicates(['epoch'] if epoch_or_step == 'epoch' else ['step'], keep = 'last')

    df.to_csv(os.path.join(log_directory, run_name + '.csv'), index = False)

def get_question_length_stats(data_directory):
    """
        Reads the preprocessed train data and computes the statistics of
        length of questions (i.e., number of words) and their frequencies.
    """
    with open(os.path.join(data_directory, 'train_data.txt'), 'r') as f:
        train_data = f.read().strip().split('\n')

    questions      = [x.split('\t')[1].strip() for x in train_data]
    words          = [x.split() for x in questions]
    lengths        = [len(x) for x in words]

    count          = collections.Counter(lengths)
    plt.bar(count.keys(), count.values())
    count          = sorted(count.items())

    df             = pd.DataFrame(count, columns=['sequence_length', 'count'])
    df['perc']     = 100 * df['count'].cumsum() / df['count'].sum()

    return df
