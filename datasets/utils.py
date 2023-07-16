import torchvision.transforms as trn
import torchvision.datasets as dset
import torch
import numpy as np


def build_dataset(dataset, noise_type, noise_rate, mode="train", data_num=50000, origin_dataset=None,
                  use_test_transform=False):
    if origin_dataset is None:
        origin_dataset = dataset

    # mean and standard deviation of channels of CIFAR-10 images
    mean, std = get_dataset_normlize(origin_dataset)

    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                       trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    if use_test_transform:
        train_transform = test_transform

    if dataset == 'cifar10':
        from datasets.cifar import CIFAR10, CIFAR100

        if mode == "train":
            data = CIFAR10(root='./data/',
                           download=True,
                           dataset_type="train",
                           transform=train_transform,
                           noise_type=noise_type,
                           noise_rate=noise_rate
                           )
        else:
            data = CIFAR10(root='./data/',
                           download=True,
                           dataset_type="test",
                           transform=test_transform,
                           noise_type=noise_type,
                           noise_rate=noise_rate
                           )
        num_classes = 10
    elif dataset == 'cifar100':
        from datasets.cifar import CIFAR10, CIFAR100
        if mode == "train":
            data = CIFAR100(root='./data/',
                            download=True,
                            dataset_type="train",
                            transform=train_transform,
                            noise_type=noise_type,
                            noise_rate=noise_rate
                            )
        else:
            data = CIFAR100(root='./data/',
                            download=True,
                            dataset_type="test",
                            transform=test_transform,
                            noise_type=noise_type,
                            noise_rate=noise_rate
                            )
        num_classes = 100


    return data, num_classes


def get_dataset_normlize(dataset):
    if dataset == "cifar100":
        mean = (0.507, 0.487, 0.441)
        std = (0.267, 0.256, 0.276)
    elif dataset == "imagenet1k":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif dataset == "clothing1m":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dataset == "webvision":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.492, 0.482, 0.446)
        std = (0.247, 0.244, 0.262)
    return mean, std


