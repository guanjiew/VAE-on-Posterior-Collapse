import torch.nn as nn

from torch.nn import init

__all__ = ['betavaeh']

from vae_model.vae_arch import *


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class BetaVAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017).
        input_dim + 2*padding - k_size)/stride + 1
    """

    def __init__(self, z_dim=10, nc=3, is_classification=False, num_classes=None):
        super(BetaVAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.vae = VAE(nc, z_dim)

        self.is_classification = is_classification
        self.num_classes = num_classes
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        # if self.is_classification:
        #     classif = self.fc(distributions)
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.vae.en_forward(x)

    def _decode(self, x):
        return self.vae.de_forward(x)


def betavaeh(**kwargs):
    return BetaVAE(**kwargs)
