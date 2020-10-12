from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .downstream import Classifier
from .utils import DomBN, BN
import torch.nn as nn


def form_models(config):
    if config.sep_bn == 1:
        print('Using domain specific BN')
        norm_layer = DomBN
    else:
        print('Using single BN')
        norm_layer = BN

    if config.model == 'resnet18':
        netF = resnet18(pretrained=True, sep_bn=(config.sep_bn == 1), norm_layer=norm_layer)
        nemb = 512
    elif config.model == 'resnet34':
        netF = resnet34(pretrained=True, sep_bn=(config.sep_bn == 1), norm_layer=norm_layer)
        nemb = 512
    elif config.model == 'resnet50':
        netF = resnet50(pretrained=True, sep_bn=(config.sep_bn == 1), norm_layer=norm_layer)
        nemb = 2048
    elif config.model == 'resnet101':
        netF = resnet101(pretrained=True, sep_bn=(config.sep_bn == 1), norm_layer=norm_layer)
        nemb = 2048
    elif config.model == 'resnet152':
        netF = resnet152(pretrained=True, sep_bn=(config.sep_bn == 1), norm_layer=norm_layer)
        nemb = 2048
    else:
        raise ValueError('Model cannot be recognized.')
    return netF, nemb
