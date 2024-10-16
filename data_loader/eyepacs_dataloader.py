from torch.utils.data import DataLoader
from torchvision import transforms

class EyePacsDataLoader(DataLoader):
    """
    Data Loader for DR Grading Score from EyePacs
    """
    def __init__(self, dataset, batch_size=16, shuffle=True, num_workers=1, training=True):
        self.shuffle = shuffle
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)

