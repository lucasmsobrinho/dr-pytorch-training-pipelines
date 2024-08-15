import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Load a pretrained ResNet model
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='DEFAULT')

        # Replace final classifier layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # Define the forward pass
        x = self.resnet(x)
        return F.log_softmax(x, dim=1)
