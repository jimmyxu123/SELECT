import torch
import timm
import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import copy
# import os
# from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from getDataLoader import get_data_loader, get_default_device, to_device
import numpy as np
from tqdm import tqdm
random_seed = 1234
torch.manual_seed(random_seed)

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, max_steps):
    current_steps = 0
    while current_steps < max_steps:
        for X, y in dataloader:
            if current_steps >= max_steps:
                break
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            current_steps += 1
            if current_steps % 100 == 0:
                print("Steps: " + str(current_steps))
    return model


def test_loop(dataloader, model):
    model.eval()
    size = len(dataloader.dl.dataset)
    correct = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    return correct #Accuracy as a decimal

def finetune_eval():
    device = get_default_device()
    LR = 0.01
    MAX_STEPS = 1000
    DECAY_STEPS = 300
    DECAY_GAMMA = 0.1
    MOMENTUM = 0.9
    BATCH_SIZE = 64
    INPUT_SIZE = 64
    DATASET = "svhn"
    #DATASET = "caltech256"
    train_dataloader, test_dataloader, NUM_CLASSES = get_data_loader(DATASET, "/scratch/wpy2004/vtab_ds", INPUT_SIZE, BATCH_SIZE)
    model = timm.create_model('resnet50', pretrained=True, pretrained_cfg = {'file': '/scratch/wpy2004/vtab_weights/in1000.pth.tar'})
    model.train()
    # for param in model.parameters():
    #     param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = to_device(model, device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = DECAY_STEPS, gamma=DECAY_GAMMA)
    finetune_model = train_loop(train_dataloader, model, loss_fn, optimizer, scheduler, MAX_STEPS)
    acc = test_loop(test_dataloader, finetune_model)
    print(acc)

if __name__ == "__main__":
    finetune_eval()
