import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, Omniglot, SVHN
import src.datasets
import numpy as np

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

def get_dataset(dataset_name, data_path, train, imgsz=28, mean=None, std=None):

    if mean is None and std is None:
        if dataset_name == "MNIST":
            mean = 0.1307
            std = 0.3081
        elif dataset_name == "FashionMNIST":
            mean = 0.2860
            std = 0.3530
        elif dataset_name == "Omniglot":
            mean = 0.9221
            std = 0.2681
        elif dataset_name == "CIFAR10":
            # mean = (0.4914, 0.4822, 0.4465)
            # std = (0.247, 0.243, 0.261)
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif dataset_name == "CIFAR100":
            mean = (0.5071, 0.4865, 0.4409)
            std = (0.2673, 0.2564, 0.2762)
        elif dataset_name == "SVHN":
            mean = (0.4377, 0.4438, 0.4728)
            std = (0.1980, 0.2010, 0.1970)
        elif dataset_name == "3DShapes":
            return datasets.Fast_3DShapes(train=train, train_frac = 470000/480000,data_folder = '..\data')
        else:
            raise ValueError(f"{dataset_name} is not supported.")

    transform = get_transform(imgsz, train, mean=mean, std=std, dataset_name=dataset_name)
    if dataset_name == "MNIST":
        return MNIST(data_path, download=True, train=train, transform=transform)
    elif dataset_name == "FashionMNIST":
        return FashionMNIST(data_path, download=True, train=train, transform=transform)
    elif dataset_name == "Omniglot":
        return Omniglot(data_path, download=True, background=train, transform=transform)
    elif dataset_name == "CIFAR10":
        return CIFAR10(data_path, download=True, train=train, transform=transform)
    elif dataset_name == "CIFAR100":
        trainset  = CIFAR100(data_path, download=True, train=train, transform=transform)
        #trainset.targets = sparse2coarse(trainset.targets)
        return trainset

    elif dataset_name == "SVHN":
        return SVHN(
            data_path,
            download=True,
            split="train" if train else "test",
            transform=transform,
        )
    elif dataset_name == "isun":
        return torchvision.datasets.ImageFolder(
            os.path.join(data_path, "iSUN"), transform=transform
        )
    elif dataset_name == "LSUN":
        return torchvision.datasets.ImageFolder(
            os.path.join(data_path, "LSUN"), transform=transform
        )
    elif dataset_name == "LSUN_resize":
        return torchvision.datasets.ImageFolder(
            os.path.join(data_path, "LSUN_resize"), transform=transform
        )
    elif dataset_name == "Textures":
        return torchvision.datasets.ImageFolder(
            os.path.join(data_path, "dtd", "images"), transform=transform
        )
    elif dataset_name == "Places365":
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_path, "Places"), transform=transform
        )
        dataset_subset = torch.utils.data.Subset(dataset, list(range(10000)))
        return dataset_subset
    else:
        raise ValueError(f"{dataset_name} is not supported.")


def get_transform(imgsz=28, train=True, mean=(0.1307), std=(0.3081), dataset_name = ""):
    if train:
        transformations = [
                transforms.Resize([imgsz, imgsz]),
                #transforms.RandomResizedCrop(size=imgsz, scale=[0.2, 1]),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(10),
                #transforms.RandAugment(2),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        #if dataset_name != "MNIST":
        #    transformations.append(transforms.Grayscale(1))      
  #  transforms.ColorJitter(brightness=0.5, hue=0.3),
        # transforms.RandomAffine(degrees=30,translate =(0.2,0.2),scale=(0.75,1.0)),
        # transforms.RandomHorizontalFlip(),
    else:
        transformations = [
                transforms.Resize([imgsz, imgsz]),
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ]
        #if dataset_name != "MNIST":
        #    transformations.append(transforms.Grayscale(1))      
    transform = transforms.Compose(transformations)
    return transform


def get_train_val_dataset(dataset_name):
    data_path = os.path.join(os.getcwd(), "..", "data")

    train = get_dataset(dataset_name, data_path=data_path, train=True)
    validation = get_dataset(dataset_name, data_path=data_path, train=False)

    return train, validation


def get_dataloader(dataset, batch_size, num_workers, train):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if train else False,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )


def get_train_val_loaders(dataset_name, batch_size=128, num_workers=10):
    train, validation = get_train_val_dataset(dataset_name)

    train_loader = get_dataloader(
        train, batch_size=batch_size, num_workers=num_workers, train=True
    )

    val_loader = get_dataloader(
        validation, batch_size=batch_size, num_workers=num_workers, train=False
    )

    return train_loader, val_loader