import torch
import torch.nn as nn

from convs.dyconv import *

__all__ = ['Dy_AlexNet']

class Dy_AlexNet(nn.Module):

    def __init__(self, num_classes=200, num_experts=3):
        super(Dy_AlexNet, self).__init__()
        self.features = nn.Sequential(
            DyConv(3, 64, kernel_size=3, stride=2, padding=1, num_experts=num_experts),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DyConv(64, 192, kernel_size=3, padding=1, num_experts=num_experts),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            DyConv(192, 384, kernel_size=3, padding=1, num_experts=num_experts),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            DyConv(384, 256, kernel_size=3, padding=1, num_experts=num_experts),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DyConv(256, 256, kernel_size=3, padding=1, num_experts=num_experts),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x