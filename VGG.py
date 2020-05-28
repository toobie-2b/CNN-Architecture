import torch
import torch.nn as nn
import torch.nn.functional as  F

CONFIG = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
	def __init__(self, config, in_channels=3, num_classes=1000):
		super(VGG, self).__init__()
		self.config = config
		self.in_channels = in_channels

		self.features = self.create_features(self.config, self.in_channels)

		self.classifier = nn.Sequential(
				nn.Flatten(),
				nn.Linear(25088, 4096),
				nn.ReLU(),
				nn.Dropout(p=0.5),
				nn.Linear(4096, 4096),
				nn.ReLU(),
				nn.Dropout(p=0.5),
				nn.Linear(4096, num_classes)
			)

	def forward(self, x):
		x = self.features(x)
		x = F.softmax(self.classifier(x), dim=1)
		return x

	def create_features(self, config, in_features):
		layers = []

		for out_features in CONFIG[config]:
			if type(out_features) == int:
				layers.append(nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1))
				layers.append(nn.ReLU())
				in_features = out_features
			else:
				layers.append(nn.MaxPool2d(2, 2))

		return nn.Sequential(*layers)