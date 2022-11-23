from copy import deepcopy
from tensorboardX import SummaryWriter

import numpy as np
import os
import time
import torch

def train(model, train_loader, loss_fn, optimizer, device, completed_steps, print_step_freq = 50, print_stats = True, step_writer = None):
    model.train()
    train_loss      = 0.0
    train_accuracy  = 0.0
    num_samples     = 0

    for step, (images, questions, answers) in enumerate(train_loader):
        images           = images.to(device)
        questions        = questions.to(device)
        answers          = answers.to(device)

        optimizer.zero_grad()

        pred_scores      = model(images, questions)
        l                = loss_fn(pred_scores, answers)
        train_loss      += l.item() * images.size(0)

        _, pred_answers  = torch.max(pred_scores, 1)
        train_accuracy  += (pred_answers == answers).sum().item()

        l.backward()
        optimizer.step()
        
        num_samples     += images.size(0)

        completed_steps += 1
        if print_stats and (completed_steps == 1 or completed_steps % print_step_freq == 0):
            print(f"Step - {completed_steps}, Running Train Loss - {train_loss/num_samples:.4f}, Running Train Accuracy - {train_accuracy * 100.0 /num_samples:.2f}")
        
        if step_writer is not None:
            step_writer.add_scalar('Train_Loss', train_loss/num_samples, completed_steps)
            step_writer.add_scalar('Train_Accuracy', train_accuracy * 100.0/num_samples, completed_steps)

    train_loss         /= num_samples
    train_accuracy      = train_accuracy * 100.0 / num_samples

    return model, optimizer, train_loss, train_accuracy

def val(model, val_loader, loss_fn, device):
    model.eval()
    val_loss     = 0.0
    val_accuracy = 0.0
    num_samples  = 0

    for step, (images, questions, answers) in enumerate(val_loader):
        images          = images.to(device)
        questions       = questions.to(device)
        answers         = answers.to(device)

        pred_scores     = model(images, questions)
        l               = loss_fn(pred_scores, answers)
        val_loss       += l.item() * images.size(0)

        _, pred_answers = torch.max(pred_scores, 1)
        val_accuracy   += (pred_answers == answers).sum().item()
        
        num_samples    += images.size(0)

    val_loss           /= num_samples
    val_accuracy        = val_accuracy * 100.0 / num_samples

    return model, val_loss, val_accuracy

def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, save_directory, log_directory,
                epochs = 50, run_name = 'testrun', save_model = False, save_best_state = True,
                print_epoch_freq = 1, print_step_freq = 50, print_stats = True):
    start_time          = time.time()
    train_length        = len(train_loader.dataset)
    train_batch_size    = train_loader.batch_size
    train_steps         = np.ceil(train_length / train_batch_size)

    best_epoch_file     = run_name + '_best.txt'
    if os.path.exists(os.path.join(save_directory, best_epoch_file)): # This is resuming training
        with open(os.path.join(save_directory, best_epoch_file), 'r') as f:
            start_epoch, best_accuracy = f.read().strip().split('\n')
        start_epoch     = int(start_epoch) + 1
        best_accuracy   = float(best_accuracy)
        model.load_state_dict(torch.load(os.path.join(save_directory, run_name + '_best.pth')))
        print(f"Starting training from epoch - {start_epoch}!")
    else:
        best_accuracy   = 0
        start_epoch     = 1

    epoch_writer        = SummaryWriter(os.path.join(log_directory, run_name))
    step_writer         = SummaryWriter(os.path.join(log_directory, run_name + "_step"))
    for epoch in range(start_epoch, epochs + 1):
        completed_steps   = (epoch - 1) * train_steps

        train_start_time  = time.time()
        model, optimizer, train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, device, completed_steps,
                                                             print_step_freq = print_step_freq, print_stats = print_stats,
                                                             step_writer = step_writer)
        train_time        = time.time() - train_start_time

        val_start_time    = time.time()
        model, val_loss, val_accuracy = val(model, val_loader, loss_fn, device)
        val_time          = time.time() - val_start_time

        if print_stats and (epoch == 1 or epoch % print_epoch_freq == 0):
            print(f'Epoch - {epoch}, Train Loss - {train_loss:.4f}, Train Accuracy - {train_accuracy:.2f}, '
                  f'Val Loss - {val_loss:.4f}, Val Accuracy - {val_accuracy:.2f}, '
                  f'Train Time - {train_time:.2f} secs, Val Time - {val_time:.2f} secs')

        if save_model:
            print(f"Saving model at epoch {epoch}!")
            torch.save(model.state_dict(), os.path.join(save_directory, run_name + '_' + str(epoch) + '.pth'))

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_weights  = deepcopy(model.state_dict())

            if save_best_state:
                print(f"Saving best model with val accuracy {val_accuracy}!")
                torch.save(model.state_dict(), os.path.join(save_directory, run_name + '_best.pth'))
                with open(os.path.join(save_directory, best_epoch_file), 'w') as out:
                    out.write(str(epoch) + "\n")
                    out.write(str(best_accuracy) + "\n")

        epoch_writer.add_scalar('Train_Loss', train_loss, epoch)
        epoch_writer.add_scalar('Train_Accuracy', train_accuracy, epoch)
        epoch_writer.add_scalar('Train_Time', train_time, epoch)
        epoch_writer.add_scalar('Val_Loss', val_loss, epoch)
        epoch_writer.add_scalar('Val_Accuracy', val_accuracy, epoch)
        epoch_writer.add_scalar('Val_Time', val_time, epoch)

        if scheduler is not None:
            scheduler.step()

    total_time = time.time() - start_time
    print(f'Best Val Accuracy - {best_accuracy}, Total Train Time - {total_time:.2f} secs')

    if save_best_state and best_accuracy > 0:
        model.load_state_dict(best_weights)

    epoch_writer.close()
    step_writer.close()

    return model, optimizer, best_accuracy

# WRITE A FUNCTION THAT TAKES AN IMAGE AND A QUESTION AND RETURNS THE ANSWER
