import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import contextlib
import torchvision.utils as vutils
from pathlib import Path

##############################################################################################
# Losses

# DCGAN loss
def loss_dcgan(dis_real, dis_fake, weights):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake) * weights)
    loss = L1 + L2
    return loss


# Wasserstein loss
def loss_wasserstein(dis_real, dis_fake, weights):
    L1 = torch.mean(-dis_real)
    L2 = torch.mean(dis_fake * weights)
    loss = L1 + L2
    return loss


# Hinge Loss
def loss_hinge(dis_real, dis_fake, weights):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake) * weights)
    loss = loss_real + loss_fake
    return loss


loss_factory = {
    'ns': loss_dcgan,
    'wasserstein': loss_wasserstein,
    'hinge': loss_hinge
}

##############################################################################################
# Penalties

def gradient_penalty(netD, real_data, fake_data, lamb, device):
    batch_size = real_data.size(0)

    assert real_data.size(0) == fake_data.size(0)

    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous().view(batch_size, -1)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return gradient_penalty


def orthogonal_regularization(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
          # Only apply this to parameters with at least 2 axes, and not in the blacklist
          if len(param.shape) < 2 or any([param is item for item in blacklist]):
            continue
          w = param.view(param.shape[0], -1)
          grad = (2 * torch.mm(torch.mm(w, w.t())
                  * (1. - torch.eye(w.shape[0], device=w.device)), w))
          param.grad.data += strength * grad.view(param.shape)


class EntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, weights=None):
        p = F.softmax(logits, dim=1)
        elementwise_entropy = -p * F.log_softmax(logits, dim=1)
        if self.reduction == 'none':
            return elementwise_entropy

        sum_entropy = torch.sum(elementwise_entropy, dim=1)
        if weights is not None:
            sum_entropy = torch.squeeze(sum_entropy) * torch.squeeze(weights)
        if self.reduction == 'sum':
            return sum_entropy

        return torch.mean(sum_entropy)


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATPerturbationGenerator(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATPerturbationGenerator, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, modelF, modelC, x):
        with torch.no_grad():
            pred = F.softmax(modelC(modelF(x)), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(modelF):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = modelC(modelF(x + self.xi * d))
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                modelF.zero_grad()
                modelC.zero_grad()

            # calc LDS
            r_adv = d * self.eps

        x_pert = x + r_adv.detach()
        return x_pert


def vat_criterion(modelF, modelC, inp, savepath='results', debug=False):
    VATGenerator = VATPerturbationGenerator(xi=0.000001, eps=1.0, ip=15)
    pert = VATGenerator(modelF, modelC, inp)

    with _disable_tracking_bn_stats(modelF):
        with torch.no_grad():
            pred = modelC(modelF(inp))

        pred_hat = modelC(modelF(pert))
        logp_hat = F.log_softmax(pred_hat, dim=1)
        lds = F.kl_div(logp_hat, pred, reduction='batchmean')

    if debug:
        Path('{}/debug'.format(savepath)).mkdir(exist_ok=True)
        print('Perturbation norm: {}'.format(torch.norm((pert - inp)[0])))
        vutils.save_image(inp, '{}/debug/orig.png'.format(savepath), normalize=True)
        vutils.save_image(pert, '{}/debug/perturbed.png'.format(savepath), normalize=True)

    return lds
