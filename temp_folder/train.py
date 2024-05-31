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
# Prof's reco
# from architecture import ResNet9, Bottleneck, BasicBlock, get_default_device, to_device
# Random kaggle
from architecture import ResNet9, get_default_device, to_device, DeviceDataLoader
from torchinfo import summary
from trainUtils import fit_one_cycle, generate_predictions, save_predictions_to_csv, evaluate, get_mean_and_std
from accuracyUtils import plot_accuracies, plot_losses, plot_lrs
from dataset import get_train_dataloader, get_test_dataloader, get_real_test_dataloader

if __name__ == "__main__":
    device = get_default_device()
    trainDataLoader, validDataLoader = get_train_dataloader()
    trainDataLoader = DeviceDataLoader(trainDataLoader, device)
    validDataLoader = DeviceDataLoader(validDataLoader, device)
    model = to_device(ResNet9(3, 10), device)
    summary(model, input_size = (400, 3, 128, 128))
    print("Trainable Parameters: "+ str(summary(model, input_size = (400, 3, 128, 128)).trainable_params))
    epochs = 800
    max_lr = 1e-3
    grad_clip = 0.12
    weight_decay = 0.00001
    #opt_func = torch.optim.Adam(model.parameters(), max_lr, amsgrad=True, weight_decay=weight_decay)
    opt_func = torch.optim.SGD(model.parameters(), lr=max_lr,
                      momentum=0.9, weight_decay=5e-4)
    history = fit_one_cycle(epochs, max_lr, model, trainDataLoader, validDataLoader, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
    plot_accuracies(history)
    plot_losses(history)
    plot_lrs(history)
    testDataLoader = get_test_dataloader()
    testDataLoader = DeviceDataLoader(testDataLoader, device)
    predictions = generate_predictions(model, testDataLoader)
    test_ids = list(range(len(predictions)))
    print("test_ids: ", test_ids[:5])
    save_predictions_to_csv(predictions, test_ids)
    torch.save(model.state_dict(), "model.pth")
    realTestDataLoader = get_real_test_dataloader()
    realTestDataLoader = DeviceDataLoader(realTestDataLoader, device)
    real_results = evaluate(model, realTestDataLoader)
    print(real_results)

    