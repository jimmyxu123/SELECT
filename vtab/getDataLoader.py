import os
import ssl
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle

from torchvision.transforms import ToTensor
from torch.utils.data import random_split, Subset, Dataset, DataLoader

random_seed = 1234
torch.manual_seed(random_seed)

ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs("vtab_ds", exist_ok=True)
os.makedirs("dataset_prep", exist_ok=True)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
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
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        inputs = inputs.to("cpu")
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def prep_dataset(root, input_size, dataset):
    temp_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )
    if dataset == "caltech256":
        temp_dataset = torchvision.datasets.Caltech256(
            root=root, transform=temp_transform, download=True
        )
        indices = np.arange(len(temp_dataset))
        grayscale_list = []
        for i in range(len(temp_dataset)):
            if temp_dataset[i][0].shape[0] == 1:
                grayscale_list.append(i)
        rgb_indices = [i for i in indices if i not in grayscale_list]
        with open("dataset_prep/caltech256_rgb.pkl", "wb") as file:
            pickle.dump(rgb_indices, file)
        rgb_dataset = Subset(temp_dataset, rgb_indices)
        test_dataset, train_dataset = random_split(
            rgb_dataset, [0.2, 0.8], generator=torch.Generator()
        )
        ds_mean, ds_std = get_mean_and_std(train_dataset)
        f = open("dataset_prep/caltech256_meanstd.txt", "a")
        print("Caltech-256 Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "svhn":
        temp_dataset = torchvision.datasets.SVHN(
            root=root, transform=temp_transform, download=True, split="train"
        )
        ds_mean, ds_std = get_mean_and_std(temp_dataset)
        f = open("dataset_prep/svhn_meanstd.txt", "a")
        print("SVHN Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "dtd":
        temp_dataset = torchvision.datasets.DTD(
            root=root, transform=temp_transform, download=True, split="train"
        )
        ds_mean, ds_std = get_mean_and_std(temp_dataset)
        f = open("dataset_prep/dtd_meanstd.txt", "a")
        print("DTD Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "eurosat":
        temp_dataset = torchvision.datasets.EuroSAT(
            root=root, transform=temp_transform, download=True
        )
        test_dataset, train_dataset = random_split(
            temp_dataset, [0.2, 0.8], generator=torch.Generator()
        )
        ds_mean, ds_std = get_mean_and_std(train_dataset)
        f = open("dataset_prep/eurosat_meanstd.txt", "a")
        print("EuroSAT Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "flowers102":
        temp_dataset = torchvision.datasets.Flowers102(
            root=root, transform=temp_transform, download=True, split="train"
        )
        ds_mean, ds_std = get_mean_and_std(temp_dataset)
        f = open("dataset_prep/flowers102_meanstd.txt", "a")
        print("Flowers102 Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "country211":
        temp_dataset = torchvision.datasets.Country211(
            root=root, transform=temp_transform, download=True, split="train"
        )
        ds_mean, ds_std = get_mean_and_std(temp_dataset)
        f = open("dataset_prep/country211_meanstd.txt", "a")
        print("Country211 Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "fgvcaircraft":
        temp_dataset = torchvision.datasets.FGVCAircraft(
            root=root, transform=temp_transform, download=True, split="train"
        )
        ds_mean, ds_std = get_mean_and_std(temp_dataset)
        f = open("dataset_prep/fgvcaircraft_meanstd.txt", "a")
        print("FGVCAircraft Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "gtsrb":
        temp_dataset = torchvision.datasets.GTSRB(
            root=root, transform=temp_transform, download=True, split="train"
        )
        ds_mean, ds_std = get_mean_and_std(temp_dataset)
        f = open("dataset_prep/gtsrb_meanstd.txt", "a")
        print("GTSRB Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "renderedsst2":
        temp_dataset = torchvision.datasets.RenderedSST2(
            root=root, transform=temp_transform, download=True, split="train"
        )
        ds_mean, ds_std = get_mean_and_std(temp_dataset)
        f = open("dataset_prep/renderedsst2_meanstd.txt", "a")
        print("RenderedSST2 Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "lfwpeople":
        temp_dataset = torchvision.datasets.LFWPeople(
            root=root, transform=temp_transform, download=True, split="train"
        )
        ds_mean, ds_std = get_mean_and_std(temp_dataset)
        f = open("dataset_prep/lfwpeople_meanstd.txt", "a")
        print("LFWPeople Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    elif dataset == "sun397":
        temp_dataset = torchvision.datasets.SUN397(
            root=root, transform=temp_transform, download=True
        )
        test_dataset, train_dataset = random_split(
            temp_dataset, [0.2, 0.8], generator=torch.Generator()
        )
        ds_mean, ds_std = get_mean_and_std(train_dataset)
        f = open("dataset_prep/sun397_meanstd.txt", "a")
        print("SUN397 Train Statistics:", file=f)
        print("Mean: " + str(ds_mean), file=f)
        print("STD: " + str(ds_std), file=f)
        f.close()

    else:
        print("Dataset prep not available.")


def get_data_loader(dataset, root, input_size, batch_size):
    # Returns train, test dataloaders for specified torchvision dataset and the # of classes
    device = get_default_device()
    if dataset == "caltech256":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5523, 0.5333, 0.5045], [0.2222, 0.2214, 0.2246]
                ),
            ]
        )
        temp_dataset = torchvision.datasets.Caltech256(
            root=root, transform=temp_transform, download=True
        )
        with open("dataset_prep/caltech256_rgb.pkl", "rb") as file:
            rgb_indices = pickle.load(file)
        rgb_dataset = Subset(temp_dataset, rgb_indices)
        test_dataset, train_dataset = random_split(
            rgb_dataset, [0.2, 0.8], generator=torch.Generator()
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            257,
        )
    elif dataset == "svhn":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4386, 0.4447, 0.4737], [0.1167, 0.1197, 0.1022]
                ),
            ]
        )
        train_dataset = torchvision.datasets.SVHN(
            root=root, transform=temp_transform, download=True, split="train"
        )
        test_dataset = torchvision.datasets.SVHN(
            root=root, transform=temp_transform, download=True, split="test"
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            10,
        )
    elif dataset == "dtd":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5288, 0.4730, 0.4247], [0.1437, 0.1437, 0.1399]
                ),
            ]
        )
        train_dataset = torchvision.datasets.DTD(
            root=root, transform=temp_transform, download=True, split="train"
        )
        test_dataset = torchvision.datasets.DTD(
            root=root, transform=temp_transform, download=True, split="test"
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            47,
        )
    elif dataset == "eurosat":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.3443, 0.3801, 0.4076], [0.0914, 0.0652, 0.0553]
                ),
            ]
        )
        temp_dataset = torchvision.datasets.EuroSAT(
            root=root, transform=temp_transform, download=True
        )
        test_dataset, train_dataset = random_split(
            temp_dataset, [0.2, 0.8], generator=torch.Generator()
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            10,
        )
    elif dataset == "flowers102":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4330, 0.3819, 0.2964], [0.2487, 0.1980, 0.2101]
                ),
            ]
        )
        train_dataset = torchvision.datasets.Flowers102(
            root=root, transform=temp_transform, download=True, split="train"
        )
        test_dataset = torchvision.datasets.Flowers102(
            root=root, transform=temp_transform, download=True, split="test"
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            102,
        )
    elif dataset == "country211":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4570, 0.4503, 0.4208], [0.2130, 0.2082, 0.2194]
                ),
            ]
        )
        train_dataset = torchvision.datasets.Country211(
            root=root, transform=temp_transform, download=True, split="train"
        )
        test_dataset = torchvision.datasets.Country211(
            root=root, transform=temp_transform, download=True, split="test"
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            211,
        )
    elif dataset == "fgvcaircraft":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4814, 0.5123, 0.5357], [0.1716, 0.1708, 0.1952]
                ),
            ]
        )
        train_dataset = torchvision.datasets.FGVCAircraft(
            root=root, transform=temp_transform, download=True, split="train"
        )
        test_dataset = torchvision.datasets.FGVCAircraft(
            root=root, transform=temp_transform, download=True, split="test"
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            100,
        )
    elif dataset == "gtsrb":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.3418, 0.3126, 0.3216], [0.1629, 0.1624, 0.1721]
                ),
            ]
        )
        train_dataset = torchvision.datasets.GTSRB(
            root=root, transform=temp_transform, download=True, split="train"
        )
        test_dataset = torchvision.datasets.GTSRB(
            root=root, transform=temp_transform, download=True, split="test"
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            43,
        )
    elif dataset == "renderedsst2":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.9847, 0.9847, 0.9847], [0.0564, 0.0564, 0.0564]
                ),
            ]
        )
        train_dataset = torchvision.datasets.RenderedSST2(
            root=root, transform=temp_transform, download=True, split="train"
        )
        test_dataset = torchvision.datasets.RenderedSST2(
            root=root, transform=temp_transform, download=True, split="test"
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            2,
        )
    elif dataset == "lfwpeople":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4334, 0.3755, 0.3333], [0.2630, 0.2362, 0.2262]
                ),
            ]
        )
        train_dataset = torchvision.datasets.LFWPeople(
            root=root, transform=temp_transform, download=True, split="train"
        )
        test_dataset = torchvision.datasets.LFWPeople(
            root=root, transform=temp_transform, download=True, split="test"
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            5749,
        )
    elif dataset == "sun397":
        temp_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4757, 0.4601, 0.4247], [0.2114, 0.2102, 0.2245]
                ),
            ]
        )
        temp_dataset = torchvision.datasets.SUN397(
            root=root, transform=temp_transform, download=True
        )
        test_dataset, train_dataset = random_split(
            temp_dataset, [0.2, 0.8], generator=torch.Generator()
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)
        return (
            DeviceDataLoader(train_dataloader, device),
            DeviceDataLoader(test_dataloader, device),
            397,
        )
    else:
        print("Dataset test not available.")


if __name__ == "__main__":
    input_size = 64
    # dataset = "caltech256"
    DATASET_LIST = [
        "caltech256",
        "svhn",
        "dtd",
        "eurosat",
        "flowers102",
        "country211",
        "fgvcaircraft",
        "gtsrb",
        "renderedsst2",
        "lfwpeople",
        "sun397",
    ]
    for dataset in DATASET_LIST:
        train, test, x = get_data_loader(dataset, "vtab_ds", input_size,
                                         input_size)
        print(len(test.dl.dataset))
    # prep_dataset("vtab_ds", input_size, dataset)
