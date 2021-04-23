__all__ = ['betavaeh']

from vae_model.vae_arch import *


class BetaVAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017).
        input_dim + 2*padding - k_size)/stride + 1
    """

    def __init__(self, z_dim=10, nc=3, is_classification=False, num_classes=None):
        super(BetaVAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            # in, out, kernel, stride, padding
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  16,  16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 2, 1),  # B, 256, 4, 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            View((-1, 256 * 4 * 4)),  # B, 256
            nn.Linear(256 * 4 * 4, z_dim * 2),  # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256 * 4 * 4),  # B, 256
            View((-1, 256, 4, 4)),  # B, 256,  4,  4
            nn.ReLU(True),
            ## stride *(n-1) + kernel - 2padding
            nn.ConvTranspose2d(256, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64, 1.e-3),
            nn.ReLU(True),
            ##
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8*2,  8*2
            nn.BatchNorm2d(64, 1.e-3),
            nn.ReLU(True),
            ##
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16*2, 16*2
            nn.BatchNorm2d(32, 1.e-3),
            nn.ReLU(True),
            ##
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 64, 64
            nn.BatchNorm2d(32, 1.e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 128, 128
        )
        self.is_classification = is_classification
        self.num_classes = num_classes
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self.decode(z)

        # if self.is_classification:
        #     classif = self.fc(distributions)
        return x_recon, mu, logvar

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def betavaeh(**kwargs):
    return BetaVAE(**kwargs)
