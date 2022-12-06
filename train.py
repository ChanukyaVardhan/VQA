from copy import deepcopy
from tensorboardX import SummaryWriter

import numpy as np
import os
import time
import torch

def train(model, train_loader, loss_fn, optimizer, device, completed_steps, initial_lr, use_sigmoid = False, use_sftmx_multiple_ans = False, print_step_freq = 50, print_stats = True, step_writer = None):
    """
        train the model for an epoch with data from train_loader on device
    """
    model.train()
    train_loss      = 0.0
    train_accuracy  = 0.0
    num_samples     = 0

    for step, (images, questions, answers, all_answers, ans_score) in enumerate(train_loader):
        images           = images.to(device)
        questions        = questions.to(device)
        answers          = answers.to(device)
        ans_score        = ans_score.to(device)

        optimizer.zero_grad()

        pred_scores      = model(images, questions)
        if use_sigmoid:
            l            = loss_fn(pred_scores, ans_score) * ans_score.size(1)
        elif use_sftmx_multiple_ans:
            l            = (-loss_fn(pred_scores) * ans_score).sum(dim = 1).mean()
        else:
            l            = loss_fn(pred_scores, answers)
        train_loss      += l.item() * images.size(0)

        _, pred_answers  = torch.max(pred_scores, 1)
        train_accuracy  += (pred_answers == answers).sum().item()

        l.backward()
        optimizer.step()
        
        num_samples     += images.size(0)
        completed_steps += 1

        # print train statistics
        if print_stats and (completed_steps == 1 or completed_steps % print_step_freq == 0):
            print(f"Step - {completed_steps}, Running Train Loss - {train_loss/num_samples:.4f}, Running Train Accuracy - {train_accuracy * 100.0 /num_samples:.2f}")
        
        # log the running train loss and running train accuracy for each step using step_writer
        if step_writer is not None:
            step_writer.add_scalar('Train_Loss', train_loss/num_samples, completed_steps)
            step_writer.add_scalar('Train_Accuracy', train_accuracy * 100.0/num_samples, completed_steps)

    train_loss         /= num_samples
    train_accuracy      = train_accuracy * 100.0 / num_samples

    return model, optimizer, train_loss, train_accuracy

def val(model, val_loader, loss_fn, device, use_sigmoid = False, use_sftmx_multiple_ans = False):
    """
        calculate the validation loss and validation accuracy and also VQA accuracy metric
    """
    model.eval()
    val_loss     = 0.0
    val_accuracy = 0.0
    vqa_accuracy = 0.0
    num_samples  = 0

    for step, (images, questions, answers, all_answers, ans_score) in enumerate(val_loader):
        images          = images.to(device)
        questions       = questions.to(device)
        answers         = answers.to(device)
        ans_score       = ans_score.to(device)

        pred_scores     = model(images, questions)
        if use_sigmoid:
            l            = loss_fn(pred_scores, ans_score) * ans_score.size(1)
        elif use_sftmx_multiple_ans:
            l            = (-loss_fn(pred_scores) * ans_score).sum(dim = 1).mean()
        else:
            l            = loss_fn(pred_scores, answers)
        val_loss       += l.item() * images.size(0)

        _, pred_answers = torch.max(pred_scores, 1)
        val_accuracy   += (pred_answers == answers).sum().item()
        all_answers[all_answers == 0] = 30000 # so we don't match answers that are not in vocabulary
        vqa_accuracy   += (np.sum(pred_answers.cpu().numpy().reshape(-1, 1).repeat(10, axis = 1) == all_answers.numpy(), axis = 1) / 3.0).clip(max = 1.0).sum()
        
        num_samples    += images.size(0)

    val_loss           /= num_samples
    val_accuracy        = val_accuracy * 100.0 / num_samples
    vqa_accuracy        = vqa_accuracy * 100.0 / num_samples

    return model, val_loss, val_accuracy, vqa_accuracy

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, save_directory, log_directory, initial_lr,
                epochs = 50, run_name = 'testrun', use_sigmoid = False, use_sftmx_multiple_ans = False,
                save_best_state = True, save_logs = True, print_epoch_freq = 1, print_step_freq = 50, print_stats = True):
    """
        - model:               model to train loaded on the device
        - train_loader:        data loader for training data
        - val_loader:          data loader for validation data
        - loss_fn:             loss function
        - optimizer:           optimizer
        - device:              device to run training on
        - save_directory:      directory to save the best model checkpoints (indexes on run_name)
        - log_directory:       directory to log training statistics at epoch and step level (indexes on run_name)
        - initial_lr:          initial learning rate, will be decayed based on number of completed iterations for adam and rmsprop
        - epochs:              number of epochs to train (this is the final_epoch number, training starts from existing best epoch)
        - run_name:            unique identifier for the experiment, all the checkpoints and logs are saved using this run_name
        - save_best_state:     saves the best performing model on validation set accuracy (this can used to resume training as well)
        - save_logs:           save logs to log_directory
        - print_epoch_freq:    print training statistics after these many epochs
        - print_step_freq:     print training statistics after these many steps
        - print_stats:         print statistics
    """
    start_time          = time.time()
    train_length        = len(train_loader.dataset)
    train_batch_size    = train_loader.batch_size
    train_steps         = np.ceil(train_length / train_batch_size)

    # If best model already exists for this experiment, continue training from the epoch after that
    best_epoch_file     = run_name + '_best.txt'
    if os.path.exists(os.path.join(save_directory, best_epoch_file)): # This is resuming training
        with open(os.path.join(save_directory, best_epoch_file), 'r') as f:
            start_epoch, best_accuracy = f.read().strip().split('\n') # read epoch number and the best accuracy
        start_epoch     = int(start_epoch) + 1
        best_accuracy   = float(best_accuracy)
        model.load_state_dict(torch.load(os.path.join(save_directory, run_name + '_best.pth'))) # load the model
        optimizer.load_state_dict(torch.load(os.path.join(save_directory, run_name + '_optimizer_best.pth'))) # load the optimizer
        best_weights    = deepcopy(model.state_dict())
    else: # start training from 1st epoch
        best_accuracy   = 0
        start_epoch     = 1
        best_weights    = None

    # update learning rate to the epoch
    completed_steps     = (start_epoch - 1) * train_steps

    print(f"Starting training from epoch - {start_epoch}!")

    if save_logs:
        epoch_writer      = SummaryWriter(os.path.join(log_directory, run_name)) # summary writer for epoch level stats
        step_writer       = SummaryWriter(os.path.join(log_directory, run_name + "_step")) # summary writer for step level stats
    for epoch in range(start_epoch, epochs + 1):
        completed_steps   = (epoch - 1) * train_steps

        train_start_time  = time.time()
        model, optimizer, train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, device, completed_steps, initial_lr,
                                                             use_sigmoid = use_sigmoid, use_sftmx_multiple_ans = use_sftmx_multiple_ans,
                                                             print_step_freq = print_step_freq, print_stats = print_stats,
                                                             step_writer = step_writer if save_logs else None)
        train_time        = time.time() - train_start_time

        val_start_time    = time.time()
        model, val_loss, val_accuracy, vqa_accuracy = val(model, val_loader, loss_fn, device, use_sigmoid = use_sigmoid, use_sftmx_multiple_ans = use_sftmx_multiple_ans)
        val_time          = time.time() - val_start_time

        # print statistics
        if print_stats and (epoch == 1 or epoch % print_epoch_freq == 0):
            print(f'Epoch - {epoch}, Train Loss - {train_loss:.4f}, Train Accuracy - {train_accuracy:.2f}, '
                  f'Val Loss - {val_loss:.4f}, Val Accuracy - {val_accuracy:.2f}, VQA Accuracy - {vqa_accuracy:.2f}, '
                  f'Train Time - {train_time:.2f} secs, Val Time - {val_time:.2f} secs')

        if vqa_accuracy > best_accuracy: # found a new best model
            best_accuracy = vqa_accuracy
            best_weights  = deepcopy(model.state_dict())

            if save_best_state: # save the best model with the epoch number and accuracy
                print(f"Saving best model with vqa accuracy {vqa_accuracy}!")
                torch.save(model.state_dict(), os.path.join(save_directory, run_name + '_best.pth'))
                torch.save(optimizer.state_dict(), os.path.join(save_directory, run_name + '_optimizer_best.pth'))
                with open(os.path.join(save_directory, best_epoch_file), 'w') as out:
                    out.write(str(epoch) + "\n")
                    out.write(str(best_accuracy) + "\n")

        # log train loss, train accuracy, train time, val loss, val accuracy, val time
        if save_logs:
            epoch_writer.add_scalar('Train_Loss', train_loss, epoch)
            epoch_writer.add_scalar('Train_Accuracy', train_accuracy, epoch)
            epoch_writer.add_scalar('Train_Time', train_time, epoch)
            epoch_writer.add_scalar('Val_Loss', val_loss, epoch)
            epoch_writer.add_scalar('Val_Accuracy', val_accuracy, epoch)
            epoch_writer.add_scalar('VQA_Accuracy', vqa_accuracy, epoch)
            epoch_writer.add_scalar('Val_Time', val_time, epoch)

    total_time = time.time() - start_time
    print(f'Best VQA Accuracy - {best_accuracy}, Total Train Time - {total_time:.2f} secs')

    if save_best_state and best_accuracy > 0: # load the model with best state
        model.load_state_dict(best_weights)

    # close log writers
    if save_logs:
        epoch_writer.close()
        step_writer.close()

    return model, optimizer, best_accuracy

def get_VQA_accuracy(model, val_loader, device):
    """
        Computes the VQA accuracy metric, answer is deemed accurate if at least 
        3 workers provided the exact answer
    """
    model.eval()
    vqa_accuracy = 0.0
    num_samples  = 0

    for step, (images, questions, answers, all_answers, ans_score) in enumerate(val_loader):
        images          = images.to(device)
        questions       = questions.to(device)
        all_answers[all_answers == 0] = 30000 # so we don't match answers that are not in vocabulary
        all_answers     = all_answers.to(device)

        pred_scores     = model(images, questions)
        _, pred_answers = torch.max(pred_scores, 1)
        
        vqa_accuracy   += (np.sum(pred_answers.cpu().numpy().reshape(-1, 1).repeat(10, axis = 1) == all_answers.cpu().numpy(), axis = 1) / 3.0).clip(max = 1.0).sum()
        num_samples    += images.size(0)

    vqa_accuracy        = vqa_accuracy * 100.0 / num_samples
    print(f'VQA Accuracy for the best model - {vqa_accuracy}')

    return model, vqa_accuracy

# WRITE A FUNCTION THAT TAKES AN IMAGE AND A QUESTION AND RETURNS THE ANSWER
