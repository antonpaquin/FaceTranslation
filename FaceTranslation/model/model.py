import torch.nn as nn

from .encoder import Encoder
from .discriminator import Discriminator
from .generator import Generator


class CombinedModel(nn.Module):
    def __init__(self, scalebias_per_step):
        super(CombinedModel, self).__init__()
        self.encoder = Encoder(scalebias_per_step)
        self.discriminator = Discriminator(scalebias_per_step)
        self.generator = Generator(scalebias_per_step)

    def generator_parameters(self):
        yield from self.encoder.parameters()
        yield from self.generator.parameters()

    def discriminator_parameters(self):
        yield from self.discriminator.parameters()

    def forward(self, x, enc_key):
        e = self.encoder(x, enc_key)
        d = self.discriminator(e)
        g = self.generator(e, enc_key)
        return g, d

