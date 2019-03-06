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


# Anton: wtf

# First, make sure disc out has nonzero gradients wrt x
# It's so accurate it's making me think that there's info being dropped somewhere
# --> there's something that G can see that D cant (besides the key)

# If that's not the case, see if there's any easy discriminator upgrades possible
# It might be that deep conv isn't the right way to solve is it anime

# Or just leave it running for a few days, or bump up disc loss weight again
# Who knows maybe it will converge
# I might have to let D train below 0.2 again

# I'd like to try some kind of morphing / distortion with the generator but that will
# take *math* (!) and still won't help without a good discriminator

# I was thinking about another D on gen vs orig after G, but it looks like StyleGAN
# and MSE are handling things just fine

# G is good enough that I might try adding color back in
# I think that would mostly help D...?

# Also maybe ask gwern


# Morphing:

# Some kind of 2d recurrent bullshit?

# Figure out the math behind grid expand / contract?


def get_x(batch):
    return torch.tensor(next(batch)).permute(0, 3, 1, 2).float()


def main():
    batch_size = 32
    num_steps = 30000

    run_timestamp = datetime.now()

    #model = CombinedModel(scalebias_per_step=2)
    model = load_model()

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
        cross_loss = (.2 * encoder_loss) + (.8 * (1 - discriminator_loss).clamp(min=0))

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
