from torch.utils.data import DataLoader
from torchvision import transforms
from base.base_data_loader import BaseDataLoader
from data_loader.ddr_dataset import DDRDataset

class DDRDataLoader(BaseDataLoader):
    """
    Data Loader for DR Grading Score from DDR
    """
    def __init__(self, batch_size=16, validation_split=0.1, shuffle=True, num_workers=1, training=True):
        self.shuffle = shuffle
        

        centercrop224 = transforms.Compose([transforms.ToPILImage(),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

        self.dataset = DDRDataset(transform=centercrop224)
        self.n_samples = len(self.dataset)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)
