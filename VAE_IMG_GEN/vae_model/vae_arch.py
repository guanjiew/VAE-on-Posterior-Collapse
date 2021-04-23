import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class VAE(nn.Module):
    def __init__(self, nc, z_dim):
        super(VAE, self).__init__()
        self.nc = nc
        self.z_dim = z_dim
        num_filters = 32
        kernel = 4

        # Encoder Architecture
        self.en_blk1 = nn.Sequential(
            nn.Conv2d(nc, num_filters, kernel_size=kernel, stride=2, padding=1),  # B, 32, 64, 64
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.en_blk2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=kernel, stride=2, padding=1),  # B, 32, 32, 32
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.en_blk3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel, stride=2, padding=1),  # B,  64,  16,  16
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU()
        )
        self.en_blk4 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=kernel, stride=2, padding=1),  # B,  64,  8,  8
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU()
        )

        self.en_blk5 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=kernel, stride=2, padding=1),  # B, 256, 4, 4
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU())

        self.en_blk6 = nn.Sequential(
            View((-1, num_filters * 4 * 4 * 4)),  # B, 256
            nn.Linear(num_filters * 4 * 4 * 4, z_dim * 2),  # B, z_dim*2
        )

        # Decoder Architecture
        self.de_blk1 = nn.Sequential(
            nn.Linear(z_dim, num_filters * 4 * 4 * 4),  # B, 256
            View((-1, num_filters * 4, 4, 4)),  # B, 256,  4,  4
            nn.ReLU(),
        )
        self.de_blk2 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=kernel, stride=2, padding=1),
            # B,  64, 8,  8
            nn.BatchNorm2d(num_filters * 2, 1.e-3),
            nn.ReLU(),
        )

        self.de_blk3 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, num_filters * 2, kernel_size=kernel, stride=2, padding=1),
            # B,  64,  8*2,  8*2
            nn.BatchNorm2d(num_filters * 2, 1.e-3),
            nn.ReLU(),
        )

        self.de_blk4 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=kernel, stride=2, padding=1),
            # B, 32, 16*2, 16*2
            nn.BatchNorm2d(num_filters, 1.e-3),
            nn.ReLU(),
        )

        self.de_blk5 = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=kernel, stride=2, padding=1),
            # B,  32, 64, 64
            nn.BatchNorm2d(num_filters, 1.e-3),
            nn.ReLU()
        )

        self.de_blk6 = nn.ConvTranspose2d(num_filters, nc, 4, 2, 1)  # B, nc, 128, 128

    def forward(self, x):
        z = self.en_forward(x)
        x_rec = self._decode(z)
        return x_rec

    def encode_z2(self,x):
        blk1 = self.en_blk1(x)
        blk2 = self.en_blk2(blk1)
        blk3 = self.en_blk3(blk2)
        blk4 = self.en_blk4(blk3)
        blk5 = self.en_blk5(blk4)
        z2 = self.en_blk6(blk5)
        return z2

    def en_forward(self, x):
        z2 = self.encode_z2(x)
        z = reparametrize(z2[:, :self.z_dim], z2[:, self.z_dim:])
        return z

    def de_forward(self, z):
        blk1 = self.de_blk1(z)
        blk2 = self.de_blk2(blk1)
        blk3 = self.de_blk3(blk2)
        blk4 = self.de_blk4(blk3)
        blk5 = self.de_blk5(blk4)
        rec_x = self.de_blk6(blk5)
        return rec_x


class Skip_VAE(VAE):
    def __init__(self, nc, z_dim):
        super(Skip_VAE, self).__init__(nc, z_dim)
        num_filters = 32
        kernel = 4

        # Decoder for skip VAE
        self.de_blk1 = nn.Sequential(
            nn.Linear(z_dim, num_filters * 4 * 4 * 4),  # B, 256
            View((-1, num_filters * 4, 4, 4)),  # B, 256,  4,  4
            nn.ReLU(),
        )
        self.de_blk2 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 8, num_filters * 2, kernel_size=kernel, stride=2, padding=1),
            # B,  64, 8,  8
            nn.BatchNorm2d(num_filters * 2, 1.e-3),
            nn.ReLU(),
        )

        self.de_blk3 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=kernel, stride=2, padding=1),
            # B,  64,  8*2,  8*2
            nn.BatchNorm2d(num_filters * 2, 1.e-3),
            nn.ReLU(),
        )

        self.de_blk4 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 4, num_filters, kernel_size=kernel, stride=2, padding=1),
            # B, 32, 16*2, 16*2
            nn.BatchNorm2d(num_filters, 1.e-3),
            nn.ReLU(),
        )

        self.de_blk5 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=kernel, stride=2, padding=1),
            # B,  32, 64, 64
            nn.BatchNorm2d(num_filters, 1.e-3),
            nn.ReLU()
        )

    def de_forward(self, x, z):
        en_blk1 = self.en_blk1(x)  # B, 32, 64, 64
        en_blk2 = self.en_blk2(en_blk1)  # B, 32, 32, 32
        en_blk3 = self.en_blk3(en_blk2)  # B, 64, 16, 16
        en_blk4 = self.en_blk4(en_blk3)  # B, 64, 8, 8
        en_blk5 = self.en_blk5(en_blk4)  # B, 128, 4, 4
        de_blk1 = self.de_blk1(z)  # B, 128, 4, 4
        de_blk2 = self.de_blk2(torch.cat([en_blk5, de_blk1], dim=1))  # B, 64, 8, 8
        de_blk3 = self.de_blk3(torch.cat([en_blk4, de_blk2], dim=1))  # B, 64, 16, 16
        de_blk4 = self.de_blk4(torch.cat([en_blk3, de_blk3], dim=1))  # B, 32, 32, 32
        de_blk5 = self.de_blk5(torch.cat([en_blk2, de_blk4], dim=1))  # B, 32, 64, 64
        recon_x = self.de_blk6(de_blk5)  # B, nc, 128, 128
        return recon_x

    def forward(self, x):
        en_blk1 = self.en_blk1(x)  # B, 32, 64, 64
        en_blk2 = self.en_blk2(en_blk1)  # B, 32, 32, 32
        en_blk3 = self.en_blk3(en_blk2)  # B, 64, 16, 16
        en_blk4 = self.en_blk4(en_blk3)  # B, 64, 8, 8
        en_blk5 = self.en_blk5(en_blk4)  # B, 128, 4, 4
        z2 = self.en_blk6(en_blk5)  # B, z_dim*2
        mu = z2[:, :self.z_dim]
        logvar = z2[:, self.z_dim:]
        z = reparametrize(mu, logvar)  # B, z_dim
        de_blk1 = self.de_blk1(z)  # B, 128, 4, 4
        de_blk2 = self.de_blk2(torch.cat([en_blk5, de_blk1], dim=1))  # B, 64, 8, 8
        de_blk3 = self.de_blk3(torch.cat([en_blk4, de_blk2], dim=1))  # B, 64, 16, 16
        de_blk4 = self.de_blk4(torch.cat([en_blk3, de_blk3], dim=1))  # B, 32, 32, 32
        de_blk5 = self.de_blk5(torch.cat([en_blk2, de_blk4], dim=1))  # B, 32, 64, 64
        x = self.de_blk6(de_blk5)  # B, nc, 128, 128
        return x
