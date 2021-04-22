import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vae_model.betavae import BetaVAE


def he_init(m):
    s = np.sqrt(2. / m.in_features)
    m.weight.data.normal_(0, s)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, in_dim, out_dim=None, kernel_size=3, mask='B'):
        super(GatedMaskedConv2d, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.dim = out_dim
        self.size = kernel_size
        self.mask = mask
        pad = self.size // 2

        # vertical stack
        self.v_conv = nn.Conv2d(in_dim, 2 * self.dim, kernel_size=(pad + 1, self.size))
        self.v_pad1 = nn.ConstantPad2d((pad, pad, pad, 0), 0)
        self.v_pad2 = nn.ConstantPad2d((0, 0, 1, 0), 0)
        self.vh_conv = nn.Conv2d(2 * self.dim, 2 * self.dim, kernel_size=1)

        # horizontal stack
        self.h_conv = nn.Conv2d(in_dim, 2 * self.dim, kernel_size=(1, pad + 1))
        self.h_pad1 = nn.ConstantPad2d((self.size // 2, 0, 0, 0), 0)
        self.h_pad2 = nn.ConstantPad2d((1, 0, 0, 0), 0)
        self.h_conv_res = nn.Conv2d(self.dim, self.dim, 1)
        self.h_res = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, v_map, h_map):
        v_out = self.v_pad2(self.v_conv(self.v_pad1(v_map)))[:, :, :-1, :]
        v_map_out = F.tanh(v_out[:, :self.dim]) * F.sigmoid(v_out[:, self.dim:])
        vh = self.vh_conv(v_out)

        h_out = self.h_conv(self.h_pad1(h_map))
        if self.mask == 'A':
            h_out = self.h_pad2(h_out)[:, :, :, :-1]
        h_out = h_out + vh
        h_out = F.tanh(h_out[:, :self.dim]) * F.sigmoid(h_out[:, self.dim:])
        h_map_out = self.h_conv_res(h_out)
        if self.mask == 'B':
            h_map_out = h_map_out + self.h_res(h_map)
        return v_map_out, h_map_out


class StackedGatedMaskedConv2d(nn.Module):
    def __init__(self,
                 img_size=[1, 28, 28], layers=[64, 64, 64],
                 kernel_size=[7, 7, 7], latent_dim=64, latent_feature_map=1, skip=0):
        super(StackedGatedMaskedConv2d, self).__init__()
        self.skip = skip
        input_dim = img_size[0]
        self.conv_layers = []
        self.z_linears = nn.ModuleList()
        if latent_feature_map > 0:
            self.latent_feature_map = latent_feature_map
        if self.skip == 0:
            add_dim = 0
        else:
            add_dim = latent_feature_map
        for i in range(len(kernel_size)):
            self.z_linears.append(nn.Linear(latent_dim, latent_feature_map * 28 * 28))
            if i == 0:
                self.conv_layers.append(GatedMaskedConv2d(input_dim + latent_feature_map,
                                                          layers[i], kernel_size[i], 'A'))
            else:
                self.conv_layers.append(GatedMaskedConv2d(layers[i - 1] + add_dim,
                                                          layers[i], kernel_size[i]))

        self.modules = nn.ModuleList(self.conv_layers)

    def forward(self, img, q_z=None):
        # if q_z is not None:
        #   z_img = self.z_linear(q_z)
        #   z_img = z_img.view(img.size(0), self.latent_feature_map, img.size(2), img.size(3))
        v_map = img
        h_map = img
        for i in range(len(self.conv_layers)):
            z_img_i = self.z_linears[i](q_z).view(img.size(0), self.latent_feature_map, 28, 28)
            if i == 0 or self.skip == 1:
                v_map = torch.cat([v_map, z_img_i], 1)
                h_map = torch.cat([h_map, z_img_i], 1)
            v_map, h_map = self.conv_layers[i](v_map, h_map)
        return h_map


class CNNVAE(BetaVAE):
    def __init__(self,
                 z_dim=10,
                 dec_kernel_size=[7, 7, 7],
                 dec_layers=[64, 64, 64],
                 latent_feature_map=4,
                 skip=0,
                 nc=3):
        super(CNNVAE, self).__init__()
        self.encoder = get_encoder(nc, z_dim)
        self.skip = skip
        self.dec_cnn = StackedGatedMaskedConv2d(img_size=img_size, layers=dec_layers,
                                                latent_dim=z_dim, kernel_size=dec_kernel_size,
                                                latent_feature_map=latent_feature_map,
                                                skip=self.skip)
        if self.skip == 0:
            self.dec_linear = nn.Conv2d(dec_layers[-1], img_size[0], kernel_size=1)
        else:
            self.dec_linear = nn.Conv2d(dec_layers[-1] + latent_feature_map, img_size[0], kernel_size=1)
        self.decoder = nn.ModuleList([self.dec_cnn, self.dec_linear])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)


def skip_vae(**kwargs):
    """
    Constructs a VAE model with skip connections.
    """
    return CNNVAE()
