import torch
import torch.nn as nn
from .layers import *
from .base import *
import utils
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.ngpu = int(config.ngpu)
        nz = int(config.nz)
        ngf = int(config.ngf)
        nc = config.nc
        self.config = config

        if config.conditional and (config.conditioning == 'concat' or config.conditioning == 'acgan'):
            inp_dim = nz + config.num_classes
        else:
            inp_dim = nz
        conv_layer = cond_conv_layers[config.G_conv]
        norm_layer = cond_norm_layers[config.G_normalization]
        activation_layer = cond_activation_layers[config.G_activation]

        init_size = int(config.imageSize / (2 ** 3))

        self.network = nn.Sequential(
            conv_layer( inp_dim, ngf * 8, init_size, 1, 0, bias=config.G_bias),
            norm_layer(ngf * 8, config.num_classes),
            activation_layer(True),
            conv_layer(ngf * 8, ngf * 4, 4, 2, 1, bias=config.G_bias),
            norm_layer(ngf * 4, config.num_classes),
            activation_layer(True),
            conv_layer(ngf * 4, ngf * 2, 4, 2, 1, bias=config.G_bias),
            norm_layer(ngf * 2, config.num_classes),
            activation_layer(True),
            conv_layer(ngf * 2,     ngf, 4, 2, 1, bias=config.G_bias),
            norm_layer(ngf, config.num_classes),
            activation_layer(True),
            conv_layer(ngf, nc, 3, 1, 1, bias=config.G_bias),
            CondTanh()
        )
        self.network.apply(utils.weights_init)

    def forward(self, input_noise, label=None):
        if self.config.conditional and (self.config.conditioning == 'concat' or self.config.conditioning == 'acgan'):
            assert label is not None
            label_onehot = utils.form_onehot(label, self.config.num_classes, device=input_noise.device)
            input_noise = torch.cat((input_noise, label_onehot), dim=1)
        input_noise = input_noise.view(input_noise.size(0), input_noise.size(1), 1, 1)
        inp = (input_noise, label)
        output = self.network(inp)
        out, _ = output
        return out


class Discriminator(BaseDiscriminator):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        self.ngpu = int(config.ngpu)
        self.config = config
        ndf = int(config.ndf)
        nc = config.nc

        conv_layer = conv_layers[config.D_conv]
        activation_layer = activation_layers[config.D_activation]
        norm_layer = norm_layers[config.D_normalization]

        out_dim = config.projection_dim
        if not config.conditional:
            out_dim = 1

        self.network = nn.Sequential(
            conv_layer(nc, ndf, 3, 1, 1, bias=config.D_bias),
            activation_layer(True),
            conv_layer(ndf, ndf * 2, 4, 2, 1, bias=config.D_bias),
            norm_layer(ndf * 2),
            activation_layer(True),
            conv_layer(ndf * 2, ndf * 4, 4, 2, 1, bias=config.D_bias),
            norm_layer(ndf * 4),
            activation_layer(True),
            conv_layer(ndf * 4, ndf * 8, 4, 2, 1, bias=config.D_bias),
            norm_layer(ndf * 8),
            activation_layer(True),
            conv_layer(ndf * 8, out_dim, 4, 1, 0, bias=config.D_bias)
        )
        self.network.apply(utils.weights_init)

    def forward(self, input, label=None):
        disc_logits = self.network(input)
        disc_logits = torch.sum(disc_logits, (2, 3))
        disc_logits = disc_logits.view(disc_logits.size(0), -1)

        if self.config.conditional:
            disc_logits = self.project(disc_logits, label)

        return disc_logits


class DiscriminatorSNGAN(BaseDiscriminator):
    def __init__(self, config):
        super(DiscriminatorSNGAN, self).__init__(config)
        self.ngpu = int(config.ngpu)
        self.config = config
        ndf = int(config.ndf)
        nc = config.nc

        conv_layer = conv_layers[config.D_conv]
        activation_layer = activation_layers[config.D_activation]

        out_dim = config.projection_dim
        if not config.conditional:
            out_dim = 1

        self.network = nn.Sequential(
            # input is (nc) x 32 x 32
            conv_layer(nc, ndf, 3, 1, 1, bias=config.D_bias),
            activation_layer(True),
            conv_layer(ndf, ndf, 4, 2, 1, bias=config.D_bias),
            activation_layer(True),
            # state size. (ndf) x 16 x 16
            conv_layer(ndf, ndf * 2, 3, 1, 1, bias=config.D_bias),
            activation_layer(True),
            conv_layer(ndf * 2, ndf * 2, 4, 2, 1, bias=config.D_bias),
            activation_layer(True),
            # state size. (ndf*2) x 8 x 8
            conv_layer(ndf * 2, ndf * 4, 3, 1, 1, bias=config.D_bias),
            activation_layer(True),
            conv_layer(ndf * 4, ndf * 4, 4, 2, 1, bias=config.D_bias),
            activation_layer(True),
            # state size. (ndf*4) x 4 x 4
            conv_layer(ndf * 4, ndf * 8, 3, 1, 1, bias=config.D_bias),
            activation_layer(True),
            conv_layer(ndf * 8, out_dim, 4, 1, 0, bias=config.D_bias)
        )
        self.network.apply(utils.weights_init)

    def forward(self, input, label=None):
        disc_logits = self.network(input)
        disc_logits = torch.sum(disc_logits, (2, 3))
        disc_logits = disc_logits.view(disc_logits.size(0), -1)

        if self.config.conditional:
            disc_logits = self.project(disc_logits, label)

        return disc_logits


class WeightNet(BaseDiscriminator):
    def __init__(self, config):
        super(WeightNet, self).__init__(config)
        self.ngpu = int(config.ngpu)
        self.config = config
        ndf = int(config.ndf)
        nc = config.nc

        out_dim = config.projection_dim
        if not config.conditional:
            out_dim = 1

        self.network = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(ndf, 2 * ndf, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2 * ndf, 4 * ndf, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4 * ndf, out_dim, 4, 1, 0, bias=True)
        )
        self.network.apply(utils.weights_init)

    def forward(self, input, label=None):
        weight_logits = self.network(input)
        weight_logits = torch.sum(weight_logits, (2, 3))
        weight_logits = weight_logits.view(weight_logits.size(0), -1)

        if self.config.conditional:
            weight_logits = self.project(weight_logits, label)

        return F.relu(weight_logits)
