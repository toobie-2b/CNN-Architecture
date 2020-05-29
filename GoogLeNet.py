import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_features, out_features, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_features, out1, red3, out3, red5, out5, outpool):
        super(Inception, self).__init__()

        self.branch1 = conv_block(in_features, out1, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
                conv_block(in_features, red3, kernel_size=1, stride=1, padding=0),
                conv_block(red3, out3, kernel_size=3, stride=1, padding=1)
            )
        self.branch3 = nn.Sequential(
                conv_block(in_features, red5, kernel_size=1, stride=1, padding=0),
                conv_block(red5, out5, kernel_size=5, stride=1, padding=2)
            )
        self.branch4 = nn.Sequential(
                nn.MaxPool2d(3, 1),
                conv_block(in_features, outpool, kernel_size=1, stride=1, padding=1)
            )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        x = torch.cat([b1, b2, b3, b4], 1)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = F.softmax(self.fc(x), dim=1)
        return x