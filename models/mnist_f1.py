import torch
import torch.nn as nn
__all__ = ['mnist_f1']


class mnist_model(nn.Module):

    def __init__(self):
        super(mnist_model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        self.regime = {
            0: {'optimizer': 'SGD', 'lr':  1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            10: {'lr': 1e-2},
            20: {'lr':  1e-3},
            30: {'lr':  1e-4}
        }

    def forward(self, inputs):
        return self.layers(inputs.view(inputs.size(0), -1))


def mnist_f1(**kwargs):
    return mnist_model()
