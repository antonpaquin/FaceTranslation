from datetime import datetime
import signal
import math

import torch

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
# I think I want to go B&W, can colorize it later.
# Identity is working pretty well, but it probably needs the adversary to create any useful features
# I probably want to train them independently until they're both good, then start the cross-loss
# I still don't know what a VAE is
# maybe read a bunch of papers on how to get autoencoders to derive deep properties
# Maybe set up a toy circles/squares example?


def get_x(batch):
    return torch.tensor(next(batch)).permute(0, 3, 1, 2).float()


def main():
    batch_size = 32
    num_steps = 300

    run_timestamp = datetime.now()

    model = load_model('2019-02-18T15:20:47')

    anime_batch = DataGen('anime').batch_stream(batch_size)
    human_batch = DataGen('human').batch_stream(batch_size)

    # Then figure out how to cross-train, consult gan best practices, etc

    enc_loss_fn = torch.nn.L1Loss(reduction='mean')
    disc_loss_fn = torch.nn.MSELoss()
    encoder_optim = torch.optim.Adam(model.encoder_params(), lr=0.001)
    adversary_optim = torch.optim.Adam(model.adversary_params(), lr=0.001)

    for step in range(num_steps):
        if (step % 2 == 0):
            x = get_x(anime_batch)
            enc_key = torch.zeros_like(x)
            adv_label = torch.Tensor((0, 1))
        else:
            x = get_x(human_batch)
            enc_key = torch.ones_like(x)
            adv_label = torch.Tensor((1, 0))

        x_pred, adv_pred = model(x, enc_key)
        enc_loss = 2 * enc_loss_fn(x_pred, x)
        adv_loss = 4 * disc_loss_fn(adv_pred, adv_label)
        cross_loss = enc_loss + (1 - adv_loss).clamp(min=0)

        report_step(step, enc_loss.item(), adv_loss.item(), cross_loss.item())
        if step % 10 == 0:
            save_model(model, run_timestamp)

        if enc_loss.item() > 0.2:
            encoder_optim.zero_grad()
            cross_loss.backward(retain_graph=True)
            encoder_optim.step()

        if adv_loss.item() > 0.5:
            adversary_optim.zero_grad()
            adv_loss.backward()
            adversary_optim.step()

        if signal_end_training:
            break


def handle_signal(signum, stack_frame):
    global signal_end_training
    signal_end_training = True


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, handle_signal)
    torch.set_num_threads(1)
    main()
