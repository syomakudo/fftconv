import sys
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def train_dataset(name,transform,download=True,data='./datasets/'):
    train_dataset = None
    if name == 'mnist':
        train_dataset = datasets.MNIST(data, train=True, download=download,
                                  transform=transform)
    elif name == 'usps':
        train_dataset = datasets.USPS(data, train=True, download=download,
                                  transform=transform)
    elif name == 'svhn':
        train_dataset = datasets.SVHN(data, split='train', download=download,
                                  transform=transform)
    print(train_dataset)
    return train_dataset


def test_dataset(name,transform,download=True,data='./datasets/'):
    test_dataset = None
    if name == 'mnist':
        test_dataset = datasets.MNIST(data, train=False, download=download,transform=transform)
    elif name == 'usps':
        test_dataset = datasets.USPS(data, train=False, download=download,transform=transform)
    elif name == 'svhn':
        test_dataset = datasets.SVHN(data, split='test', download=download,transform=transform)

    print(test_dataset)
    return test_dataset
