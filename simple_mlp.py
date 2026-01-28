import torch.nn as nn
import torch

class SimpleMLP(nn.Module):
    def __init__(self, input_shape=(1,28,28), hidden_sizes=(256, 128), num_classes=10):
        super().__init__()
        c, h, w = input_shape
        in_features = c * h * w
        layers = []
        last = in_features
        for hs in hidden_sizes:
            layers.append(nn.Linear(last, hs))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
            last = hs
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
