
from base.base_data_loader import BaseDataLoader
import numpy as np
import torchvision

class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, train=True, download=True, transform=None):
        super().__init__(root='.data', train=train, download=download, transform=transform)


class CIFAR10DataLoader(BaseDataLoader):
    """
    Data Loader for CIFAR 10
    """
    def __init__(self, dataset=None, batch_size=16, validation_split=0.1, shuffle=True, num_workers=1, training=True):
        self.shuffle = shuffle
        self.dataset = dataset

        self.n_samples = len(self.dataset)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)
