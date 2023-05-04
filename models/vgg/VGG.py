import torch
import torch.nn as nn
from torchvision import models


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super(Block, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VGG(nn.Module):

    def __init__(self, features, num_classes=7, dropout=0.5):
        super(VGG, self).__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # FIXME: max pool
        # self.pool = nn.AdaptiveMaxPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(4096, num_classes)  # like the last layer of ResNet

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x


def make_features(block_config):
    layers = []
    in_channels = 3
    for out_channels, num_layers in block_config:
        layers.append(Block(in_channels, out_channels, num_layers))
        in_channels = out_channels
    return nn.Sequential(*layers)


configs = {
    'vgg16': [2, 2, 3, 3, 3],
    'vgg19': [2, 2, 4, 4, 4],
}
conv_sizes = [64, 128, 256, 512, 512]


def vgg16(num_classes):
    # print("Using VGG16")
    # m = models.vgg16(num_classes=num_classes, pretrained=False)
    m = VGG(make_features(list(zip(conv_sizes, configs['vgg16']))), num_classes=num_classes)
    return m


def vgg19(num_classes):
    return VGG(make_features(list(zip(conv_sizes, configs['vgg19']))), num_classes=num_classes)
