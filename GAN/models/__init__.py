from . import DCGAN
from . import resnet

discriminator_factory = {
    'DCGAN': DCGAN.Discriminator,
    'SNDCGAN': DCGAN.DiscriminatorSNGAN,
    'resnet': resnet.Discriminator
}

generator_factory = {
    'DCGAN': DCGAN.Generator,
    'resnet': resnet.Generator
}

weight_factory = {
    'DCGAN': DCGAN.WeightNet,
    'SNDCGAN': DCGAN.WeightNet,
    'resnet': resnet.WeightNet
}
