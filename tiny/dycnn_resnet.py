import torch
import torch.nn as nn
import torch.nn.functional as F

from convs.dyconv import *

__all__ = ['Dy_ResNet18']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, num_experts=3):
        super().__init__()
        self.conv1 = DyConv(in_channels, channels, kernel_size=3, stride=stride, padding=1, num_experts=num_experts)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = DyConv(channels, channels, kernel_size=3, stride=1, padding=1, num_experts=num_experts)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Addition
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Dy_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200, num_experts=3):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, num_experts=num_experts)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, num_experts=num_experts)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, num_experts=num_experts)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, num_experts=num_experts)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride, num_experts):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride, num_experts))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def Dy_ResNet18(num_experts=3):
    return Dy_ResNet(BasicBlock, [2, 2, 2, 2], num_experts=num_experts)