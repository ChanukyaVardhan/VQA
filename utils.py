"""
Utility functions:
-> parse tensorboard event logs and save into csvs
-> generate statistics of length of questions
-> plot training and validation statistics
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
    vqa_accuracies    = []

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
            vqa_accuracies   += ea.Scalars('VQA_Accuracy')

    train_losses          = pd.DataFrame(train_losses).rename(columns = {'value': 'train_loss'})
    train_accuracies      = pd.DataFrame(train_accuracies).rename(columns = {'value': 'train_accuracy'})
    if epoch_or_step == 'epoch':
        val_losses            = pd.DataFrame(val_losses).rename(columns = {'value': 'val_loss'})
        val_accuracies        = pd.DataFrame(val_accuracies).rename(columns = {'value': 'val_accuracy'})
        vqa_accuracies        = pd.DataFrame(vqa_accuracies).rename(columns = {'value': 'vqa_accuracy'})

    df                    = train_losses
    df['train_accuracy']  = train_accuracies['train_accuracy']
    if epoch_or_step == 'epoch':
        df['val_loss']        = val_losses['val_loss']
        df['val_accuracy']    = val_accuracies['val_accuracy']
        df['vqa_accuracy']    = vqa_accuracies['vqa_accuracy']
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

def plot_train_val_stats(log_directory, run_name, epoch_or_step = 'epoch'):
    """
        Given log directory and the expirement run name. plots the train loss, accuracy
        and validation loss, accuracy by reading the csv files parse_tb_logs() generates
    """
    if epoch_or_step == 'step':
        run_name   += '_step'
    df              = pd.read_csv(os.path.join(log_directory, run_name + '.csv'))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # Loss plot
    plt.sca(axs[0])
    plt.plot(df[epoch_or_step].values, df["train_loss"].values, label = "Train Loss")
    if epoch_or_step == 'epoch':
        plt.plot(df[epoch_or_step].values, df["val_loss"].values, label = "Val Loss")
    plt.xlabel("Number of Epochs" if epoch_or_step == 'epoch' else "Number of Steps")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.sca(axs[1])
    plt.plot(df[epoch_or_step].values, df["train_accuracy"].values, label = "Train Accuracy")
    if epoch_or_step == 'epoch':
        plt.plot(df[epoch_or_step].values, df["val_accuracy"].values, label = "Val Accuracy")
    plt.xlabel("Number of Epochs" if epoch_or_step == 'epoch' else "Number of Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_vqa_accuracy(log_directory, run_name):
    """
        Given run_name, plots the vqa accuracy for each epoch
    """
    df              = pd.read_csv(os.path.join(log_directory, run_name + '.csv'))

    plt.plot(df['epoch'].values, df["vqa_accuracy"].values, label = "VQA Accuracy")
    plt.xlabel("Number of Epochs")
    plt.ylabel("VQA Accuracy")
    plt.legend()
    plt.show()
