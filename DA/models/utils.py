import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# Class definition for Domain specific batch norm
class DomBN(nn.Module):
    def __init__(self, nplanes, num_domains=2):
        super(DomBN, self).__init__()

        self.bn_layers = torch.nn.ModuleList()
        for i in range(num_domains):
            self.bn_layers.append(nn.BatchNorm2d(nplanes))

    def forward(self, inp):
        x = inp[0]
        dom_id = inp[1]
        return self.bn_layers[dom_id](x)


class BN(nn.Module):
    def __init__(self, nplanes):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm2d(nplanes)

    def forward(self, inp):
        x = inp[0]
        return self.bn(x)


class ConvWithIndex(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, bias=False):
        super(ConvWithIndex, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, inp):
        x = inp[0]
        dom_id = inp[1]
        return (self.conv(x), dom_id)