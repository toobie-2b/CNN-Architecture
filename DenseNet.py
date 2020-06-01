import torch
import torch.nn as nn
import torch.nn.functional as F

config = {
    'DenseNet121': 
    {
        'layers':[6, 12, 24, 16],
        'k': 32,
    },

    'DenseNet161': 
    {
        'layers':[6, 12, 36, 24],
        'k': 32,
    },

    'DenseNet169': 
    {
        'layers':[6, 12, 32, 32],
        'k': 48,
    },

    'DenseNet201': 
    {
        'layers':[6, 12, 48, 32],
        'k': 32,
    },
}


class conv_block(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(conv_block, self).__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, out_features, **kwargs)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_features, growth_rate):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_block(in_features, 128, kernel_size=1, stride=1, padding=0)
        self.conv2 = conv_block(128, growth_rate, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.cat([x, out], 1)


class DenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers):
        super(DenseLayer, self).__init__()
        self.denselayer = self.make_layers(in_features, growth_rate, num_layers)

    def forward(self, x):
        x = self.denselayer(x)
        return x

    def make_layers(self, in_features, growth_rate, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_features + i*growth_rate, growth_rate))
        return nn.Sequential(*layers)


class Transition(nn.Module):
    def __init__(self, in_features, out_features):
        super(Transition, self).__init__()
        self.transition = nn.Sequential(
                conv_block(in_features, out_features, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(2, 2)
            )

    def forward(self, x):
        x = self.transition(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, model, in_channels=3, num_classes=1000):
        super(DenseNet, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, 2, padding=1)

        self.denseblock_1 = DenseLayer(64, config[model]['k'], config[model]['layers'][0])
        self.transition_1 = Transition(64+(config[model]['layers'][0]*config[model]['k']), 128)

        self.denseblock_2 = DenseLayer(128, config[model]['k'], config[model]['layers'][1])
        self.transition_2 = Transition(128+(config[model]['layers'][1]*config[model]['k']), 256)

        self.denseblock_3 = DenseLayer(256, config[model]['k'], config[model]['layers'][2])
        self.transition_3 = Transition(256+(config[model]['layers'][2]*config[model]['k']), 512)

        self.denseblock_4 = DenseLayer(512, config[model]['k'], config[model]['layers'][3])
        self.transition_4 = Transition(512+(config[model]['layers'][3]*config[model]['k']), 1024)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.pool(self.conv(F.relu(self.bn(x))))
        x = self.denseblock_1(x)
        x = self.transition_1(x)
        x = self.denseblock_2(x)
        x = self.transition_2(x)
        x = self.denseblock_3(x)
        x = self.transition_3(x)
        x = self.denseblock_4(x)
        x = self.transition_4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.softmax(self.fc(x), dim=1)
        return x