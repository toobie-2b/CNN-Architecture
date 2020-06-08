import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_features, out_features, **kwargs)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class Stem(nn.Module):
    def __init__(self, in_features):
        super(Stem, self).__init__()
        
        self.start = nn.Sequential(
            conv_block(in_features, 32, kernel_size=3, stride=2, padding=0),
            conv_block(32, 32, kernel_size=3, stride=1, padding=0),
            conv_block(32, 64, kernel_size=3, stride=1, padding=1),
        )

        self.pool1 = nn.MaxPool2d(3, 2, padding=0)
        self.conv1 = conv_block(64, 96, kernel_size=3, stride=2, padding=0)

        self.branch1 = nn.Sequential(
                conv_block(160, 64, kernel_size=1, stride=1, padding=0),
                conv_block(64, 96, kernel_size=3, stride=1, padding=0),
            )

        self.branch2 = nn.Sequential(
                conv_block(160, 64, kernel_size=1, stride=1, padding=0),
                conv_block(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                conv_block(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                conv_block(64, 96, kernel_size=3, stride=1, padding=0),
            )

        self.pool2 = nn.MaxPool2d(3, 2, padding=0)
        self.conv2 = conv_block(192, 192, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.start(x)
        out1 = self.pool1(x)
        out2 = self.conv1(x)
        x = torch.cat([out1, out2], 1)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        x = torch.cat([out1, out2], 1)
        out1 = self.pool2(x)
        out2 = self.conv2(x)
        x = torch.cat([out1, out2], 1)
        return x


class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()

        self.branch1 = nn.Sequential(
                nn.AvgPool2d(3, 1, padding=1),
                conv_block(384, 96, kernel_size=1, stride=1, padding=0),
            )

        self.branch2 = conv_block(384, 96, kernel_size=1, stride=1, padding=0)

        self.branch3 = nn.Sequential(
                conv_block(384, 64, kernel_size=1, stride=1, padding=0),
                conv_block(64, 96, kernel_size=3, stride=1, padding=1),
            )

        self.branch4 = nn.Sequential(
                conv_block(384, 64, kernel_size=1, stride=1, padding=0),
                conv_block(64, 96, kernel_size=3, stride=1, padding=1),
                conv_block(96, 96, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x


class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()

        self.branch1 = nn.MaxPool2d(3, 2, padding=0)

        self.branch2 = conv_block(384, 384, kernel_size=3, stride=2, padding=0)

        self.branch3 = nn.Sequential(
                conv_block(384, 192, kernel_size=1, stride=1, padding=0),
                conv_block(192, 224, kernel_size=3, stride=1, padding=1),
                conv_block(224, 256, kernel_size=3, stride=2, padding=0)
            )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        x = torch.cat([branch1, branch2, branch3], 1)
        return x


class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()

        self.branch1 = nn.Sequential(
                nn.AvgPool2d(3, 1, padding=1),
                conv_block(1024, 128, kernel_size=1, stride=1, padding=0)
            )

        self.branch2 = conv_block(1024, 384, kernel_size=1, stride=1, padding=0)

        self.branch3 = nn.Sequential(
                conv_block(1024, 192, kernel_size=1, stride=1, padding=0),
                conv_block(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                conv_block(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            )

        self.branch4 = nn.Sequential(
                conv_block(1024, 192, kernel_size=1, stride=1, padding=0),
                conv_block(192, 192, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                conv_block(192, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                conv_block(224, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                conv_block(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x


class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()

        self.branch1 = nn.MaxPool2d(3, 2, padding=0)

        self.branch2 = nn.Sequential(
                conv_block(1024, 192, kernel_size=1, stride=1, padding=0),
                conv_block(192, 192, kernel_size=3, stride=2, padding=0),
            )

        self.branch3 = nn.Sequential(
                conv_block(1024, 256, kernel_size=1, stride=1, padding=0),
                conv_block(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                conv_block(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                conv_block(320, 320, kernel_size=3, stride=2, padding=0)
            )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        x = torch.cat([branch1, branch2, branch3], 1)
        return x


class InceptionC(nn.Module):
    def __init__(self):
        super(InceptionC, self).__init__()

        self.branch1 = nn.Sequential(
                nn.AvgPool2d(3, 1, padding=1),
                conv_block(1536, 256, kernel_size=1, stride=1, padding=0),
            )

        self.branch2 = conv_block(1536, 256, kernel_size=1, stride=1, padding=0)

        self.branch3_1 = conv_block(1536, 384, kernel_size=1, stride=1, padding=0)
        self.branch3_2a = conv_block(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch3_2b = conv_block(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch4_1 = nn.Sequential(
                conv_block(1536, 384, kernel_size=1, stride=1, padding=0),
                conv_block(384, 448, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                conv_block(448, 512, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            )
        self.branch4_2a = conv_block(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch4_2b = conv_block(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = self.branch3_1(x)
        out1 = self.branch3_2a(out)
        out2 = self.branch3_2b(out)
        branch3 = torch.cat([out1, out2], 1)
        out = self.branch4_1(x)
        out1 = self.branch4_2a(out)
        out2 = self.branch4_2b(out)
        branch4 = torch.cat([out1, out2], 1)
        x = torch.cat([branch1, branch2, branch3, branch4], 1)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLeNet, self).__init__()

        self.stem = Stem(in_channels)
        self.inceptionAx4 = self._make_layers(InceptionA, 4)
        self.reductionA = ReductionA()
        self.inceptionBx7 = self._make_layers(InceptionB, 7)
        self.reductionB = ReductionB()
        self.inceptionCx3 = self._make_layers(InceptionC, 3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inceptionAx4(x)
        x = self.reductionA(x)
        x = self.inceptionBx7(x)
        x = self.reductionB(x)
        x = self.inceptionCx3(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)

    def _make_layers(self, block, num_layers):
        layers = [block() for _ in range(num_layers)]
        return nn.Sequential(*layers)
