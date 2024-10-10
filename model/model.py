import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MnistModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class VGG_Jabbar(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Load the pretrained VGG model
        self.vgg = models.vgg16(weights='DEFAULT')
        for parameter in self.vgg.features.parameters():
            parameter.requires_grad = False

        # Replace final classifier layer
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)
        
    def forward(self, x):
        x = self.vgg(x)
        return x

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

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Load a pretrained ConvNeXt base model
        self.convnext = models.convnext_base(weights='DEFAULT')
        
        # Replace final classifier layer
        self.convnext.classifier[2] = nn.Linear(self.convnext.classifier[2].in_features, num_classes)

    def forward(self, x):
        x = self.convnext(x)
        return x
