import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        # Load the pretrained VGG model
        self.vgg = models.vgg16(weights='DEFAULT')
        
        # Replace output dimension with number of classes
        in_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True))

    def forward(self, x):
        x = self.vgg(x)
        return x
