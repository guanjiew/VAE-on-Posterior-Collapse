__all__ = ['skipvae']

from vae_model.vae_arch import *


class Con_VAE(nn.Module):
    def __init__(self, z_dim=10, nc=3, is_classification=False, num_classes=None):
        super(Con_VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.vae = Skip_VAE(nc, z_dim)
        self.is_classification = is_classification
        self.num_classes = num_classes
        self.encoder = self._encode
        self.decoder = self.decode

    def forward(self, x):
        distributions = self.vae.encode_z2(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        x_recon = self._decode(x)
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.vae.encode_z2(x)

    def _decode(self, x):
        return self.vae.forward(x)

    def encode(self, x):
        return self.vae.en_forward(x)

    def decode(self, x, z):
        return self.vae.de_forward(x, z)



def skipvae(**kwargs):
    return Con_VAE(**kwargs)
