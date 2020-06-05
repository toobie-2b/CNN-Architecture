import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(SeparableConv, self).__init__()

        self.spatial = nn.Conv2d(in_features, in_features, groups=in_features, **kwargs)
        self.pointwise = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.spatial(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Block, self).__init__()

        self.sep_conv1 = SeparableConv(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sep_conv2 = SeparableConv(out_features, out_features, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(3, 2, padding=1)
        self.skip = nn.Conv2d(in_features, out_features, kernel_size=1, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        original = x
        x = self.bn(self.sep_conv1(x))
        x = self.relu(x)
        x = self.bn(self.sep_conv2(x))
        x = self.pool(x)
        original = self.bn(self.skip(original))
        x += original
        return x


class MiddleBlock(nn.Module):
    def __init__(self):
        super(MiddleBlock, self).__init__()

        self.sep_conv1 = SeparableConv(728, 728, kernel_size=3, stride=1, padding=1)
        self.sep_conv2 = SeparableConv(728, 728, kernel_size=3, stride=1, padding=1)
        self.sep_conv3 = SeparableConv(728, 728, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(728)
        self.relu = nn.ReLU()

    def forward(self, x):
        original = x
        x = self.relu(self.bn(self.sep_conv1(x)))
        x = self.relu(self.bn(self.sep_conv2(x)))
        x = self.relu(self.bn(self.sep_conv3(x)))
        x += original
        return x


class Xception(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(Xception, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.entry_flow = nn.Sequential(
                Block(64, 128),
                Block(128, 256),
                Block(256, 728),
            )

        self.middle_flow = self.make_middle_flow()

        self.exit_flow = nn.Sequential(
                Block(728, 1024),
                SeparableConv(1024, 1536, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                SeparableConv(1536, 2048, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)


    def make_middle_flow(self):
        layers = []

        for _ in range(8):
            layers.append(MiddleBlock())

        return nn.Sequential(*layers)
        