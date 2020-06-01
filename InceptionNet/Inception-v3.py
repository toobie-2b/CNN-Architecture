import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_features, out_features, **kwargs)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class Grid(nn.Module):
    def __init__(self, in_features, red):
        super(Grid, self).__init__()

        self.branch1 = nn.Sequential(
                conv_block(in_features, red, kernel_size=1, stride=1, padding=0),
                conv_block(red, red, kernel_size=3, stride=1, padding=1),
                conv_block(red, red, kernel_size=3, stride=2, padding=0),
            )

        self.branch2 = nn.Sequential(
                conv_block(in_features, red, kernel_size=1, stride=1, padding=0),
                conv_block(red, red, kernel_size=3, stride=2, padding=0),
            )

        self.branch3 = nn.MaxPool2d(3, 2, padding=0)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        x = torch.cat([b1, b2, b3], 1)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, red5, out5, red3, out3, outpool, out1):
        super(InceptionA, self).__init__()

        self.branch1 = nn.Sequential(
                conv_block(in_channels, red5, kernel_size=1, stride=1, padding=0),
                conv_block(red5, out5, kernel_size=3, stride=1, padding=1),
                conv_block(out5, out5, kernel_size=3, stride=1, padding=1),
            )

        self.branch2 = nn.Sequential(
                conv_block(in_channels, red3, kernel_size=1, stride=1, padding=0),
                conv_block(red3, out3, kernel_size=3, stride=1, padding=1),
            )

        self.branch3 = nn.Sequential(
                nn.MaxPool2d(3, 1, padding=1),
                conv_block(in_channels, outpool, kernel_size=1, stride=1, padding=0),
            )

        self.branch4 = conv_block(in_channels, out1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        x = torch.cat([b1, b2, b3, b4], 1)
        return x


class InceptionB(nn.Module):
    def __init__(self, in_channels, red7_2, out7_2, red7, out7, outpool, out1):
      super(InceptionB, self).__init__()

      self.branch1 = nn.Sequential(
            conv_block(in_channels, red7_2, kernel_size=1, stride=1, padding=0),
            conv_block(red7_2, out7_2, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            conv_block(out7_2, out7_2, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            conv_block(out7_2, out7_2, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            conv_block(out7_2, out7_2, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )

      self.branch2 = nn.Sequential(
            conv_block(in_channels, red7, kernel_size=1, stride=1, padding=0),
            conv_block(red7, out7, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            conv_block(out7, out7, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

      self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            conv_block(in_channels, outpool, kernel_size=1, stride=1, padding=0)
        )

      self.branch4 = conv_block(in_channels, out1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        x = torch.cat([b1, b2, b3, b4], 1)
        return x


class InceptionC(nn.Module):
    def __init__(self, in_channels, redb1, outb1_1, outb1_1_1, outb1_1_2, redb2, outb2_1, outb2_2, outpool, out1):
        super(InceptionC, self).__init__()

        self.base1 = conv_block(in_channels, redb1, kernel_size=1, stride=1, padding=0)
        self.inter1 = conv_block(redb1, outb1_1, kernel_size=3, stride=1, padding=1)
        self.final1_1 = conv_block(outb1_1, outb1_1_1, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.final1_2 = conv_block(outb1_1, outb1_1_2, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.base2 = conv_block(in_channels, redb2, kernel_size=1, stride=1, padding=0)
        self.final2_1 = conv_block(redb2, outb2_1, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.final2_2 = conv_block(redb2, outb2_2, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
                nn.MaxPool2d(3, 1, padding=1),
                conv_block(in_channels, outpool, kernel_size=1, stride=1, padding=0)
            )

        self.branch4 = conv_block(in_channels, out1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b1 = self.base1(x)
        b1 = self.inter1(b1)
        b1_1 = self.final1_1(b1)
        b1_2 = self.final1_2(b1)

        b2 = self.base2(x)
        b2_1 = self.final2_1(b2)
        b2_2 = self.final2_2(b2)

        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1_1, b1_2, b2_1, b2_2, b3, b4], 1)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()

        self.start = nn.Sequential(
                conv_block(in_channels, 32, kernel_size=3, stride=2, padding=0),
                conv_block(32, 32, kernel_size=3, stride=1, padding=0),
                conv_block(32, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(3, 2, padding=1),
                conv_block(64, 80, kernel_size=3, stride=1, padding=0),
                conv_block(80, 192, kernel_size=3, stride=2, padding=0),
                conv_block(192, 288, kernel_size=3, stride=1, padding=1)
            )

        self.Inception1_1 = InceptionA(288, 16, 32, 96, 128, 32, 64)
        self.Inception1_2 = InceptionA(256, 32, 96, 128, 192, 64, 128)
        self.Inception1_3 = InceptionA(480, 32, 64, 128, 160, 64, 96)

        self.grid1 = Grid(384,192)

        self.Inception2_1 = InceptionB(768, 16, 48, 96, 208, 64, 192)
        self.Inception2_2 = InceptionB(512, 24, 64, 112, 224, 64, 160)
        self.Inception2_3 = InceptionB(512, 24, 64, 128, 256, 64, 128)
        self.Inception2_4 = InceptionB(512, 32, 64, 144, 288, 64, 112)
        self.Inception2_5 = InceptionB(528, 32, 80, 160, 272, 80, 208)

        self.grid2 = Grid(640,320)

        self.Inception3_1 = InceptionC(1280, 32, 64, 64, 64, 160, 160, 160, 128, 256)
        self.Inception3_2 = InceptionC(832, 48, 64, 128, 128, 192, 384, 384, 256, 768)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.start(x)
        x = self.Inception1_1(x)
        x = self.Inception1_2(x)
        x = self.Inception1_3(x)
        x = self.grid1(x)
        x = self.Inception2_1(x)
        x = self.Inception2_2(x)
        x = self.Inception2_3(x)
        x = self.Inception2_4(x)
        x = self.Inception2_5(x)
        x = self.grid2(x)
        x = self.Inception3_1(x)
        x = self.Inception3_2(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.softmax(self.fc(x), dim=1)
        return x