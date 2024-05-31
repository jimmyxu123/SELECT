import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from architecture import get_default_device

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        inputs = inputs.to('cpu')
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,weight_decay=0, 
        grad_clip=None, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func
    #sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    for epoch in (range(epochs)):
        # Training Phase 
        model.train()
        train_losses = []
        train_accuracy= []
        lrs=[]
        for (batch_idx, batch) in enumerate(train_loader):
            loss,accuracy = model.training_step(batch)
            train_losses.append(loss)
            train_accuracy.append(accuracy)
            loss.backward()
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
            if batch_idx % 60 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.4f}'.
                format(epoch+1, batch_idx , len(train_loader),
                       100. * batch_idx / len(train_loader), loss,accuracy))
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_accuracy'] = torch.stack(train_accuracy).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def generate_predictions(model, test_loader):
    device = get_default_device()  
    model.eval()  
    predictions = []
    # ids = []
    test_losses = []
    test_accs = []
    with torch.no_grad():
        for batch in test_loader:
            images, image_ids = batch 
            images, image_ids = images.to(device), image_ids.to(device)
            outputs = model(images)
            # loss = F.cross_entropy(outputs, image_ids)
            # test_losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1)
            # Convert predictions and ids to CPU before using numpy
            predictions.extend(preds.cpu().numpy())  
            # ids.extend(image_ids.cpu().numpy()) 
            #correct = torch.sum(preds == image_ids)
            #acc = correct.item() / len(preds)
            #test_accs.append(acc)
    # avg_loss = np.mean(test_losses)
    #avg_acc = np.mean(test_accs)
    #print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}')
    #print("first 5 preds and dis: ", predictions[:5])
    return predictions
