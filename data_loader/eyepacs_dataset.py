import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch

class EyePacsDataset(Dataset):
    def __init__(self, annotations_file="/home/miqueias/Eyepacs/train.csv", img_dir="/home/miqueias/Eyepacs/train/", transform=None, target_transform=None):
        annotation_doc = pd.read_csv(annotations_file)
        self.img_names = annotation_doc['image']
        self.img_labels = annotation_doc['level']
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names.iloc[idx])
        img_path += ".jpeg"
        image = read_image(img_path)
        # to float;  0-255 int -> 0, 1 float
        image = image.to(dtype=torch.float32)/255
        label = self.img_labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label