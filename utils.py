from tensorboard.backend.event_processing import event_accumulator
import os
import pandas as pd

def parse_tb_logs(log_directory, model_name, epoch_or_step = 'epoch'):
    if epoch_or_step == 'step':
        model_name   += '_step'

    train_losses      = []
    train_accuracies  = []
    val_losses        = []
    val_accuracies    = []

    directory         = os.path.join(log_directory, model_name)
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
    df.to_csv(os.path.join(log_directory, model_name + '.csv'))
