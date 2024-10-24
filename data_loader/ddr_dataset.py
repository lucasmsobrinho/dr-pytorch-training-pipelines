import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch

class DDRDataset(Dataset):
    def __init__(self, annotations_file="/home/miqueias/DDR-dataset/DR_grading/train.txt", img_dir="/home/miqueias/DDR-dataset/DR_grading/train/", transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, delimiter=" ")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        # to float;  0-255 int -> 0, 1 float
        image = image.to(dtype=torch.float32)/255
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if label >= 5:
            label = 4
        return image, label