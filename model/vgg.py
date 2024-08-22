import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        # Load the pretrained VGG model
        self.vgg = models.vgg16(weights='DEFAULT')
        
        # Replace output dimension with number of classes
        self.vgg.classifier[6].out_features = num_classes
        
    def forward(self, x):
        x = self.vgg(x)
        return nn.softmax(x, dim=1)
