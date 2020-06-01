import torch
import torch.nn as nn
import torch.nn.functional as F


config = {
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
    'ResNet152': [3, 8, 36, 3],
}

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.conv3 = nn.Conv2d(out_features, out_features*4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_features*4)
        self.downsample = downsample

    def forward(self, x):
        original = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            original = self.downsample(original)

        x += original
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, model, img_channels=3, num_classes=1000):
        super(ResNet, self).__init__()
        self.model = model
        self.layers_list = config[model]
        self.in_channels = 64
        self.conv = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        self.layer1 = self.make_res_layer(64, 1, self.layers_list[0])
        self.layer2 = self.make_res_layer(128, 2, self.layers_list[1])
        self.layer3 = self.make_res_layer(256, 2, self.layers_list[2])
        self.layer4 = self.make_res_layer(512, 2, self.layers_list[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn(self.conv(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.softmax(self.fc(x), dim=1)
        return x

    def make_res_layer(self, features, stride, num_layers):
        downsample = None
        layers=[]
        
        if stride!=1 or self.in_channels != features*4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, features*4, kernel_size=1, stride=stride, padding=0),
                    nn.BatchNorm2d(features*4)
                )
        layers.append(ResidualBlock(self.in_channels, features, stride, downsample=downsample))
        self.in_channels = features*4

        for _ in range(num_layers-1):
            layers.append(ResidualBlock(self.in_channels, features, 1))
        return nn.Sequential(*layers)
