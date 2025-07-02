import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """Classic LeNet-5 variant for 28x28 grayscale images (MNIST)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))            # [N, 6, 24, 24]
        x = F.max_pool2d(x, 2)              # [N, 6, 12, 12]
        x = F.relu(self.conv2(x))           # [N, 16, 8, 8]
        x = F.max_pool2d(x, 2)              # [N, 16, 4, 4]
        x = torch.flatten(x, 1)             # [N, 16*4*4]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_model(cfg):
    return LeNet(num_classes=cfg.model.num_classes) 