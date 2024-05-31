import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, Subset, Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
random_seed = 1234
torch.manual_seed(random_seed)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        inputs = inputs.to('cpu')
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def prep_caltech256(root, input_size):
    #Find the mean/std of training set. Also find grayscale indices and remove from dataset.
    temp_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    temp_dataset = torchvision.datasets.Caltech256(root=root, transform=temp_transform, download=True)
    indices = np.arange(len(temp_dataset))
    grayscale_list = []
    for i in range(len(temp_dataset)):
        if temp_dataset[i][0].shape[0] == 1:
            grayscale_list.append(i)
    rgb_indices = [i for i in indices if i not in grayscale_list]
    with open('dataset_prep/caltech256_rgb.pkl', 'wb') as file:
        pickle.dump(rgb_indices, file)
    rgb_dataset = Subset(temp_dataset, rgb_indices)
    test_dataset, train_dataset = random_split(rgb_dataset, [0.2, 0.8], generator=torch.Generator())
    ds_mean, ds_std = get_mean_and_std(train_dataset)
    f = open("dataset_prep/caltech256_meanstd.txt", "a")
    print("Caltech-256 Train Statistics:", file=f)
    print("Mean: " + str(ds_mean), file=f)
    print("STD: " + str(ds_std), file=f)
    f.close()

def prep_svhn(root, input_size):
    #Find the mean/std of training set.
    temp_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    temp_dataset = torchvision.datasets.SVHN(root=root, transform=temp_transform, download=True, split='train')
    ds_mean, ds_std = get_mean_and_std(temp_dataset)
    f = open("dataset_prep/svhn_meanstd.txt", "a")
    print("SVHN Train Statistics:", file=f)
    print("Mean: " + str(ds_mean), file=f)
    print("STD: " + str(ds_std), file=f)
    f.close()

def get_data_loader(dataset, root, input_size, batch_size):
    #Returns train, test dataloaders for specified torchvision dataset and the # of classes
    device = get_default_device()
    if dataset == "caltech256":
        temp_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5523, 0.5333, 0.5045], [0.2222, 0.2214, 0.2246]),
        ])
        temp_dataset = torchvision.datasets.Caltech256(root=root, transform=temp_transform, download=True)
        with open('dataset_prep/caltech256_rgb.pkl', 'rb') as file:
            rgb_indices = pickle.load(file)
        rgb_dataset = Subset(temp_dataset, rgb_indices)
        test_dataset, train_dataset = random_split(rgb_dataset, [0.2, 0.8], generator=torch.Generator())
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return DeviceDataLoader(train_dataloader, device), DeviceDataLoader(test_dataloader, device), 257
    elif dataset == "svhn":
        temp_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4386, 0.4447, 0.4737], [0.1167, 0.1197, 0.1022]),
        ])
        train_dataset = torchvision.datasets.SVHN(root=root, transform=temp_transform, download=True, split='train')
        test_dataset = torchvision.datasets.SVHN(root=root, transform=temp_transform, download=True, split='test')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return DeviceDataLoader(train_dataloader, device), DeviceDataLoader(test_dataloader, device), 10
        print()
    else:
        print("Dataset test not available.")

if __name__ == "__main__":
    input_size = 128
    #prep_caltech256("/scratch/wpy2004/vtab_ds", input_size)
    prep_svhn("/scratch/wpy2004/vtab_ds", input_size)
