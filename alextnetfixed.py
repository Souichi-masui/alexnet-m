import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNetfixed(nn.Module):
 
    def __init__(self, num_classes=10):
        super(AlexNetfixed, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True), 
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x
 