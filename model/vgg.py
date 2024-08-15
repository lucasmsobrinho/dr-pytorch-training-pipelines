import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        # Load the pretrained VGG model
        self.vgg = models.vgg16(weights='DEFAULT')
        
        # Replace final classifier layer
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)
        
    def forward(self, x):
        x = self.vgg(x)
        return x
