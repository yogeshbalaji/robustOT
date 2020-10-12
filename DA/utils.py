import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class ConfigMapper(object):
    def __init__(self, args):
        for key in args:
            self.__dict__[key] = args[key]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


class InvLR(_LRScheduler):
    """Decays the learning rate accroding to inv lr schedule
    """
    def __init__(self, optimizer, gamma=0.0001, power=0.75, last_epoch=-1):
        self.gamma = gamma
        self.power = power
        super(InvLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        factor = ((1 + self.gamma * self.last_epoch) / (1 + self.gamma * (self.last_epoch-1))) ** (-self.power)
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * factor
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * ((1 + self.gamma * self.last_epoch) ** (-self.power))
                for base_lr in self.base_lrs]


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

