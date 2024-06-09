import torch
import timm
import torch.nn as nn
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

def finetune_eval(model, finetune_layer = "fc", LR = 0.01, MAX_STEPS = 1000, DECAY_STEPS = 300, DECAY_GAMMA = 0.1, MOMENTUM = 0.9, BATCH_SIZE = 64, INPUT_SIZE = 64, DATASET = "svhn"):
    device = get_default_device()
    train_dataloader, test_dataloader, NUM_CLASSES = get_data_loader(DATASET, "vtab_ds", INPUT_SIZE, BATCH_SIZE)
    model.train()
    setattr(model, finetune_layer, nn.Linear(getattr(model, finetune_layer).in_features, NUM_CLASSES))
    model = to_device(model, device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = DECAY_STEPS, gamma=DECAY_GAMMA)
    finetune_model = train_loop(train_dataloader, model, loss_fn, optimizer, scheduler, MAX_STEPS)
    acc = test_loop(test_dataloader, finetune_model)
    test_set_n = len(test_dataloader.dl.dataset)
    SE = np.sqrt((acc * (1-acc))/test_set_n)
    CI = 1.96 * SE
    return acc, CI

if __name__ == "__main__":
    dataset = "sun397"
    model = timm.create_model('resnet50', pretrained=True)
    #model = timm.create_model('resnet50', pretrained=True, pretrained_cfg = {'file': 'vtab_weights/sd1000-t2i.tar'})
    acc = finetune_eval(model, finetune_layer = "fc", LR = 0.01, MAX_STEPS = 1000, DECAY_STEPS = 300, DECAY_GAMMA = 0.1, MOMENTUM = 0.9, BATCH_SIZE = 64, INPUT_SIZE = 64, DATASET = dataset)
    print(acc)
