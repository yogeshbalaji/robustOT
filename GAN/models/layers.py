import torch
import torch.nn as nn


###############################################################################
# Unconditional layers

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_layer = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups,
                 bias=bias, padding_mode=padding_mode))

    def forward(self, input):
        out = self.conv_layer(input)
        return out


class SpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_layer = torch.nn.utils.spectral_norm(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, input):
        out = self.linear_layer(input)
        return out


class LeakyReLU2(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=inplace)

    def forward(self, input):
        out = self.lrelu(input)
        return out


conv_layers = {
    'conv': nn.Conv2d,
    'spectral_conv': SpectralConv2d
}

linear_layers = {
    'linear': nn.Linear,
    'spectral_linear': SpectralLinear
}

activation_layers = {
    'relu': nn.ReLU,
    'lrelu': LeakyReLU2
}

norm_layers = {
    'BN': nn.BatchNorm2d,
    'identity': nn.Identity
}

###############################################################################
# Layers with conditional support


class CategoricalConditionalBatchNorm(torch.nn.Module):
    # as in the chainer SN-GAN implementation, we keep per-cat weight and bias
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, inp):
        input = inp[0]
        cats = inp[1]
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            weight = self.weight.index_select(0, cats).view(shape)
            bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return (out, cats)

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class CondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups,
                 bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.conv_layer(img)
        return (out, label)


class CondConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super().__init__()
        self.conv_transpose_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, output_padding=output_padding, groups=groups, bias=bias,
                 dilation=dilation, padding_mode=padding_mode)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.conv_transpose_layer(img)
        return (out, label)


class UnconditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes=10, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                 track_running_stats=track_running_stats)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.batch_norm(img)
        return (out, label)


class CondReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.relu(img)
        return (out, label)


class CondLeakyReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=inplace)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.lrelu(img)
        return (out, label)


class CondTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.tanh(img)
        return (out, label)


class CondUpsample(nn.Module):
    def __init__(self, scale_factor=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, input):
        img = input[0]
        label = input[1]
        out = self.upsample(img)
        return (out, label)


class CondSpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_layer = torch.nn.utils.spectral_norm(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, input):
        img, label = input
        out = self.linear_layer(img)
        return out, label


class CondLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        img, label = input
        out = self.linear_layer(img)
        return out, label


cond_norm_layers = {
    'BN': UnconditionalBatchNorm2d,
    'conditionalBN': CategoricalConditionalBatchNorm
}

cond_conv_layers = {
    'conv': CondConv2d,
    'convT': CondConvTranspose2d,
    'spectral_conv': None,
    'spectral_convT': None
}

cond_activation_layers = {
    'relu': CondReLU,
    'lrelu': CondLeakyReLU,
    'tanh': CondTanh
}

cond_linear_layers = {
    'linear': CondLinear,
    'spectral_linear': CondSpectralLinear
}
