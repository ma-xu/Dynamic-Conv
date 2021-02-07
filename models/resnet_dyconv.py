"""
The implementation of dy_conv2d is borrowed from
https://github.com/kaijieshi7/Dynamic-convolution-Pytorch/blob/master/dynamic_conv.py
Updated: I slightly re-organized the implementation for fast training,
which could be totally converted to the original implementation.

For temperature, we didn't consider the temperature annealing for easy implementation and set temperature to 34.
Moreover, introducing the annealing only imporves the accuracy by 0.5 reported in paper Table 6.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F




__all__ = ['dyconv_resnet18', 'dyconv_resnet34', 'dyconv_resnet50', 'dyconv_resnet101',
           'dyconv_resnet152']
           
class DyConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros', ratio=0.25, temperature=34):
        super(DyConv, self).__init__()
        self.temperature = temperature
        # default 3 experts for fair comparison.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                               dilation, groups, bias, padding_mode)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                               dilation, groups, bias, padding_mode)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                               dilation, groups, bias, padding_mode)
        hidden_planes = int(in_channels * ratio) + 1 if in_channels!=3 else 3
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_planes, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_planes, 3, bias=True)
        )

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1).squeeze(dim=-1).squeeze(dim=-1)  # [b,c]
        router = self.fc(gap)/self.temperature  # [b,c]
        router = torch.softmax(router, dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)  # [b,3,1,1,1]
        out1 = self.conv1(x).unsqueeze(dim=1)  # [b,1,c,w,h]
        out2 = self.conv2(x).unsqueeze(dim=1)  # [b,1,c,w,h]
        out3 = self.conv3(x).unsqueeze(dim=1)  # [b,1,c,w,h]
        out = torch.cat([out1, out2, out3], dim=1)
        out = (out*router).sum(dim=1, keepdim=False)  # [b,c,w,h]
        return out




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return DyConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return DyConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def dyconv_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def dyconv_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def dyconv_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def dyconv_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def dyconv_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def demo():
    net = dyconv_resnet50(num_classes=1000)
    y = net(torch.randn(2, 3, 224,224))
    print(y.size())


# demo()

