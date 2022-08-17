import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from Utils import seed_worker


class Dataset:
    def __init__(self, root, batch_size, train_set_size, val_set_size, *, seed=None):
        self.root = root
        self.batch_size = batch_size
        self.train_set_size = train_set_size
        self.val_set_size = val_set_size

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.g = torch.Generator()
        if seed is not None:
            self.seed_everything(seed)

    def seed_everything(self, seed):
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        self.g.manual_seed(seed)

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader

    def get_labels(self):
        return self.labels


class MNIST(Dataset):
    def __init__(self, root, batch_size, train_set_size, val_set_size, *, seed=None):
        super().__init__(root, batch_size, train_set_size, val_set_size, seed=seed)

        # Get datasets
        train_dataset = torchvision.datasets.MNIST(
            root=self.root, train=True, download=True, transform=self.transform)
        test_dataset = torchvision.datasets.MNIST(
            root=self.root, train=False, download=True, transform=self.transform)

        # Split train and validation sets
        train_set, val_set = torch.utils.data.random_split(
            train_dataset, [self.train_set_size, self.val_set_size])

        # Load train, validation, and test sets
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=self.g)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        # Index to label
        self.labels = [str(x) for x in range(10)]


class FashionMNIST(Dataset):
    def __init__(self, root, batch_size, train_set_size, val_set_size, *, seed=None):
        super().__init__(root, batch_size, train_set_size, val_set_size, seed=seed)

        # Get datasets
        train_dataset = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, download=True, transform=self.transform)
        test_dataset = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, download=True, transform=self.transform)

        # Split train and validation sets
        train_set, val_set = torch.utils.data.random_split(
            train_dataset, [self.train_set_size, self.val_set_size])

        # Load train, validation, and test sets
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=self.g)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        # Index to label
        self.labels = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class Omniglot(Dataset):
    def __init__(self, root, batch_size, train_set_size, val_set_size, *, seed=None):
        super().__init__(root, batch_size, train_set_size, val_set_size, seed=seed)

        # Get datasets
        train_dataset = torchvision.datasets.Omniglot(
            root=self.root, background=True, download=True, transform=self.transform)
        test_dataset = torchvision.datasets.Omniglot(
            root=self.root, background=False, download=True, transform=self.transform)

        # Split train and validation sets
        train_set, val_set = torch.utils.data.random_split(
            train_dataset, [self.train_set_size, self.val_set_size])

        # Load train, validation, and test sets
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=self.g)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        # Index to label
        self.labels = [str(x) for x in range(1623)]


class CIFAR10(Dataset):
    def __init__(self, root, batch_size, train_set_size, val_set_size, *, seed=None):
        super().__init__(root, batch_size, train_set_size, val_set_size, seed=seed)

        # Get datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=self.transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=self.transform)

        # Split train and validation sets
        train_set, val_set = torch.utils.data.random_split(
            train_dataset, [self.train_set_size, self.val_set_size])

        # Load train, validation, and test sets
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=self.g)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        # Index to label
        self.labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
