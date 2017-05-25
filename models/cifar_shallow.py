import torch
import torch.nn as nn
__all__ = ['cifar10_shallow', 'cifar100_shallow']


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 384, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(384, 192, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(192, num_classes)
        )
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-3,
                'weight_decay': 5e-4},
            60: {'lr': 1e-2},
            120: {'lr': 1e-3},
            180: {'lr': 1e-4}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)
        return x


def cifar10_shallow(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 10)
    return AlexNet(num_classes)


def cifar100_shallow(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 100)
    return AlexNet(num_classes)
