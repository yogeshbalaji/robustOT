import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, nemb, nclasses, nlayers, use_spectral=False):
        super(Classifier, self).__init__()

        nhidden = 256
        self.net = nn.Sequential()

        for layer in range(nlayers):
            n_in = nhidden
            n_out = nhidden
            if layer == 0:
                n_in = nemb
            if layer == nlayers-1:
                n_out = nclasses

            if use_spectral:
                self.net.add_module('fc (spectral): {}'.format(layer), 
                                    torch.nn.utils.spectral_norm(nn.Linear(n_in, n_out)))
            else:
                self.net.add_module('fc:{}'.format(layer),
                                    nn.Linear(n_in, n_out))
            if layer != nlayers-1:
                self.net.add_module('Relu: {}'.format(layer),
                                    nn.ReLU())

    def forward(self, x):
        out = self.net(x)
        return out
