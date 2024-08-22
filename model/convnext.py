import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Load a pretrained ConvNeXt base model
        self.convnext = models.convnext_base(weights='DEFAULT')
        
        # Replace final classifier layer
        self.convnext.classifier[2].out_features = num_classes

    def forward(self, x):
        x = self.convnext(x)
        return x
