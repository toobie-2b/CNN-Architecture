import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
	def __init__(self, in_channels=3, num_classes=1000):
		super(AlexNet, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0)
		self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
		self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
		self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

		self.pool = nn.MaxPool2d(3, 2)

		self.fc1 = nn.Linear(9216, 9216)
		self.fc2 = nn.Linear(9216, 4096)
		self.fc3 = nn.Linear(4096, 4096)
		self.fc4 = nn.Linear(4096, num_classes)


	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool(F.relu(self.conv5(x)))
		x = x.view(x.shape[0], -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.softmax(self.fc4(x), dim=1)
		return x