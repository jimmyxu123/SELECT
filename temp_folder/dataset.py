import os
import pickle
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from architecture import get_default_device
from customTensorDataset import CustomTensorDataset, get_transform

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_dataset(split, augment = False):
    device = get_default_device()
    if split == "train":
        cifar10_dir = 'cifar-10-python/cifar-10-batches-py'
        meta_data_dict = load_cifar_batch(os.path.join(cifar10_dir, 'batches.meta'))
        label_names = meta_data_dict[b'label_names']
        all_images = []
        all_labels = []
        for i in range(1, 6):
            batch_dict = load_cifar_batch(os.path.join(cifar10_dir, f'data_batch_{i}'))
            batch_images = batch_dict[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 1, 2, 3)
            batch_labels = batch_dict[b'labels']
            all_images.append(batch_images)
            all_labels.append(batch_labels)
        train_images_tensor = torch.Tensor(np.concatenate(all_images, axis=0)).to(device)
        train_labels_tensor = torch.Tensor(np.concatenate(all_labels, axis=0)).to(torch.long).to(device)
        X_train, X_test, y_train, y_test = train_test_split(train_images_tensor, train_labels_tensor, test_size=0.05, random_state=42)
        return X_train, X_test, y_train, y_test
    elif split == "test":
        # print("creating test dataset")
        # cifar10_dir = 'cifar-10-python/cifar-10-batches-py'
        # meta_data_dict = load_cifar_batch(os.path.join(cifar10_dir, 'batches.meta'))
        # label_names = meta_data_dict[b'label_names']
        all_images = []
        all_labels = []
        # batch_dict = load_cifar_batch(os.path.join(cifar10_dir, 'test_batch'))
        batch_dict = test_unpickle('cifar_test_nolabels.pkl')
        batch_images = batch_dict[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 1, 2, 3)
        batch_labels = batch_dict[b'ids']
        all_images.append(batch_images)
        all_labels.append(batch_labels)
        test_images_tensor = torch.Tensor(np.concatenate(all_images, axis=0)).to(device)
        test_labels_tensor = torch.Tensor(np.concatenate(all_labels, axis=0)).to(torch.long).to(device)
        return test_images_tensor, test_labels_tensor
    elif split == "real_cifar":
        cifar10_dir = 'cifar-10-python/cifar-10-batches-py'
        meta_data_dict = load_cifar_batch(os.path.join(cifar10_dir, 'batches.meta'))
        label_names = meta_data_dict[b'label_names']
        all_images = []
        all_labels = []
        batch_dict = load_cifar_batch(os.path.join(cifar10_dir, 'test_batch'))
        batch_images = batch_dict[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 1, 2, 3)
        batch_labels = batch_dict[b'labels']
        all_images.append(batch_images)
        all_labels.append(batch_labels)
        real_test_images_tensor = torch.Tensor(np.concatenate(all_images, axis=0)).to(device)
        real_test_labels_tensor = torch.Tensor(np.concatenate(all_labels, axis=0)).to(torch.long).to(device)
        return real_test_images_tensor, real_test_labels_tensor
    else:
        print("wrong input")

def get_train_dataloader():
    device = get_default_device()
    train_data, valid_data, train_labels, valid_labels = load_dataset("train", augment = True)
    train_dataset = CustomTensorDataset(tensors=(train_data, train_labels), transform=get_transform("train"))
    # train_dataset.transforms = transform_train
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = CustomTensorDataset(tensors=(valid_data, valid_labels), transform=get_transform("valid"))
    # valid_dataset.transforms = transform_valid
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

def get_test_dataloader():
    print("printing test dataloader")
    device = get_default_device()
    test_data, test_labels = load_dataset("test", augment = False)
    test_dataset = CustomTensorDataset(tensors=(test_data, test_labels), transform = get_transform("test"))
    batch_size = 400
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def get_real_test_dataloader():
    print("printing actual cifar10 test dataloader")
    device = get_default_device()
    test_data, test_labels = load_dataset("real_cifar", augment=False)
    test_dataset = CustomTensorDataset(tensors=(test_data, test_labels), transform = get_transform("test"))
    batch_size = 400
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def test_unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        temp_dict = pickle.load(fo, encoding='bytes')
    return temp_dict

# get_train_dataloader()

# if __name__ == "__main__":
#     train_data, train_labels = load_dataset("train")
#     train_dataset = TensorDataset(train_data, train_labels)
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     print(len(train_loader))
#     print(type(train_loader))
    # for images, labels in train_loader:
    #     print(images.shape)
    #     print(labels.shape)
if __name__ == "__main__":
    temp_dict = test_unpickle('cifar_test_nolabels.pkl')