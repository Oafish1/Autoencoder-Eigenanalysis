import torch
import torch.nn as nn


class Verbose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x


class PeriodicPadding2D(nn.Module):
    def __init__(self, pad_size, even_bias=True):
        """
        pad_size: (right, left, bottom, top) pad size.  Or int.
        """
        super().__init__()

        if isinstance(pad_size, int):
            if even_bias:
                self.pad_size = 2*[pad_size-1, pad_size]
            else:
                self.pad_size = 4*[pad_size]
        else:
            self.pad_size = pad_size

    def forward(self, x):
        # Bias right and bottom for even pads
        ps = self.pad_size
        x = torch.cat([x, x[:, :, :ps[0]]], dim=2)
        x = torch.cat([x[:, :, -2*ps[1]:-ps[1]], x], dim=2)
        x = torch.cat([x, x[:, :, :, :ps[2]]], dim=3)
        x = torch.cat([x[:, :, :, -2*ps[3]:-ps[3]], x], dim=3)
        return x


class AEModel(nn.Module):
    def __init__(self, input_shape=(128, 128), embedded_dim=10):
        super().__init__()

        self.encoder = nn.Sequential(
            # These dims are wonky because of even kernel size
            PeriodicPadding2D(4),
            nn.Conv2d(1, 128, 8),
            nn.MaxPool2d(2),
            nn.ReLU(),

            PeriodicPadding2D(4),
            nn.Conv2d(128, 64, 8),
            nn.MaxPool2d(2),
            nn.ReLU(),

            PeriodicPadding2D(2),
            nn.Conv2d(64, 32, 4),
            nn.MaxPool2d(2),
            nn.ReLU(),

            PeriodicPadding2D(2),
            nn.Conv2d(32, 16, 4),
            nn.MaxPool2d(2),
            nn.ReLU(),

            PeriodicPadding2D(1),
            nn.Conv2d(16, 8, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),

            # FCL
            nn.Flatten(),
            nn.Linear(128, embedded_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedded_dim, 128),
            nn.ReLU(),
            nn.Unflatten(1, (8, 4, 4)),

            # Convolutions
            nn.Upsample(scale_factor=2),
            PeriodicPadding2D(1),
            nn.Conv2d(8, 16, 2),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            PeriodicPadding2D(2),
            nn.Conv2d(16, 32, 4),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            PeriodicPadding2D(2),
            nn.Conv2d(32, 64, 4),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            PeriodicPadding2D(4),
            nn.Conv2d(64, 128, 8),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            PeriodicPadding2D(4),
            nn.Conv2d(128, 1, 8),
            nn.Tanh(),
        )

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return e, d
