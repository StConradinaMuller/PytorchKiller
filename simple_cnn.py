import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # Small, easy-to-follow CNN
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # same
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 or 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7 or 8x8
        )
        # compute flatten size generically
        dummy = torch.zeros(1, in_channels, 28 if in_channels==1 else 32, 28 if in_channels==1 else 32)
        flatten_size = self.features(dummy).view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
