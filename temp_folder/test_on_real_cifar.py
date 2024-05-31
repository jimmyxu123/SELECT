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
    model = to_device(ResNet9(3, 10), device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    realTestDataLoader = get_real_test_dataloader()
    realTestDataLoader = DeviceDataLoader(realTestDataLoader, device)
    real_results = evaluate(model, realTestDataLoader)
    print(real_results)