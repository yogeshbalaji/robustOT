import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


##############################################################################################
# Losses

# DCGAN loss
def loss_dcgan_dis(dis_real, dis_fake, weights):
    L1 = torch.mean(F.softplus(-dis_real * weights))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2


def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss


def loss_dcgan_weights(dis_real, weights):
    loss = torch.mean(F.softplus(dis_real * weights))
    return loss


# Wasserstein loss
def loss_wasserstein_dis(dis_real, dis_fake, weights):
    L1 = torch.mean(-dis_real * weights)
    L2 = torch.mean(dis_fake)
    return L1, L2


def loss_wasserstein_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def loss_wasserstein_weights(dis_real, weights):
    loss = torch.mean(dis_real * weights)
    return loss


# Hinge Loss
def loss_hinge_dis(dis_real, dis_fake, weights):
    loss_real = torch.mean(F.relu(1. - (dis_real * weights)))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def loss_hinge_weights(dis_real, weights):
    loss = -1 * torch.mean(F.relu(1. - (dis_real * weights)))
    return loss


def aux_loss(logits, labels):
    loss = F.cross_entropy(logits, labels)
    return loss


loss_factory = {
    'ns': [loss_dcgan_dis, loss_dcgan_gen],
    'wasserstein': [loss_wasserstein_dis, loss_wasserstein_gen],
    'hinge': [loss_hinge_dis, loss_hinge_gen]
}

loss_factory_weights = {
    'ns': loss_dcgan_weights,
    'wasserstein': loss_wasserstein_weights,
    'hinge': loss_hinge_weights
}

##############################################################################################
# Penalties

def gradient_penalty(netD, real_data, fake_data, lamb, device, labels=None):
    batch_size = real_data.size(0)

    assert real_data.size(0) == fake_data.size(0)

    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous().view(batch_size, 3, real_data.shape[2], real_data.shape[3])

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, labels)

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
