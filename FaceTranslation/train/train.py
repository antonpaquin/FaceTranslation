from datetime import datetime
import signal
import math

import torch
import numpy as np

from FaceTranslation.model import CombinedModel, save_model, load_model
from FaceTranslation.data.data_generator import DataGen


signal_end_training = False


def translate_score(x):
    prior = 0.5
    x_logodds = math.log(x / (1-x), 2)
    prior_logodds = math.log(prior / (1-prior), 2)
    return x_logodds - prior_logodds


def report_step(step, enc_loss, adv_loss, cross_loss):
    print('Step: {}, Autoencoder loss: {:.3f}, Discriminator loss: {:.3f}, Cross loss: {:.3f}'.format(
        step,
        enc_loss,
        adv_loss,
        cross_loss,
    ))


# Anton:
# Initial results indicate that there's something there, though it still is rough around the edges.
# Reading up on BicycleGAN, cross-domain disentanglement, StyleGAN was pretty useful.

# Turns out the bluriness is caused by loss fn issues, for which the only solution is adversarial loss
# --> I need to stick a discriminator on the output and train until the model isn't sure which is real

# StyleGAN turns out to be just a fancy kind of generator, which should be applicable here.
# It does a kind of thing where it starts with a small deep constant representation and then grows it while adding
# noise, and the latent space vector is a set of (scale, bias) pairs that are applied to the net twice per growth step
# e.g.
#    const
#    + noise
#   (z-transform) * scale + bias z1 (this layer is called AdaIN)
#    conv
#    AdaIN (z2)
#    upscale
#    + noise
#    AdaIN (z3)
#    ...
# I still don't know the exact form of Z -- is it a single constant per layer or does it have an (x,y) component?
# --> (1x1x2) or (dim x dim x 2), not sure

# Had an idea for the discriminator -- UNet seems to be pretty popular, but the easy way to do a discriminator is to
# have a single ndarray input and if you do UNet and capture the intermediate space you end up with some horrible
# unconcatable mess of ((8x8x512), (16x16x256), (32x32x128), ...)
# Soln: we're already doing (conv conv relu pool), maybe just concat the relevant layer when the decoding gets to the
# right size

# Still need to read the TwinGAN paper, results look not nearly as good as the StyleGAN ones but maybe it can be adapted


def get_x(batch):
    return torch.tensor(next(batch)).permute(0, 3, 1, 2).float()


def main():
    batch_size = 32
    num_steps = 3000

    run_timestamp = datetime.now()

    model = CombinedModel(scalebias_per_step=2)

    anime_batch = DataGen('anime').batch_stream(batch_size)
    human_batch = DataGen('human').batch_stream(batch_size)

    # Then figure out how to cross-train, consult gan best practices, etc

    enc_loss_fn = torch.nn.MSELoss()
    disc_loss_fn = torch.nn.MSELoss()
    encoder_optim = torch.optim.Adam(model.generator_parameters(), lr=0.001)
    discriminator_optim = torch.optim.Adam(model.discriminator_parameters(), lr=0.001)

    def count_params(params):
        model_parameters = filter(lambda p: p.requires_grad, params)
        return sum([np.prod(p.size()) for p in model_parameters])

    print('Trainable parameters: generator: {}, discriminator: {}'.format(
        count_params(model.generator_parameters()),
        count_params(model.discriminator_parameters()),
    ))

    for step in range(1, num_steps):
        if (step % 2 == 0):
            x = get_x(anime_batch)
            enc_key = torch.zeros((batch_size,))
            adv_label = torch.Tensor((0, 1))
        else:
            x = get_x(human_batch)
            enc_key = torch.ones((batch_size,))
            adv_label = torch.Tensor((1, 0))

        g, d = model(x, enc_key)

        encoder_loss = (3 * enc_loss_fn(g, x))
        discriminator_loss = (3 * disc_loss_fn(d, adv_label))
        cross_loss = (.5 * encoder_loss) + (.5 * (1 - discriminator_loss).clamp(min=0))

        report_step(step, encoder_loss.item(), discriminator_loss.item(), cross_loss.item())
        if step % 10 == 0:
            save_model(model, run_timestamp)

        if (1 - discriminator_loss.item()) > 0.2:
            encoder_optim.zero_grad()
            cross_loss.backward(retain_graph=True)
            encoder_optim.step()
        else:
            encoder_optim.zero_grad()
            encoder_loss.backward(retain_graph=True)
            encoder_optim.step()

        if discriminator_loss.item() > 0.2:
            discriminator_optim.zero_grad()
            discriminator_loss.backward()
            discriminator_optim.step()

        if signal_end_training:
            break


def handle_signal(signum, stack_frame):
    global signal_end_training
    signal_end_training = True


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, handle_signal)
    torch.set_num_threads(1)
    main()
