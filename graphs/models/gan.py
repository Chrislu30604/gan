import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
            nn.Linear(1024, 128 * (self.input_size // 4)
                      * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4)
                           * (self.input_size // 4)),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.RELU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1)
            nn.Tanh()
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


class Discriminator(nn.Module):

    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        