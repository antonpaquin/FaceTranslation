from datetime import datetime
import signal
import math
import os

import torch

from FaceTranslation.model import FaceTranslationModel, save_model
from FaceTranslation.data.data_generator import DataGen
from FaceTranslation.vis.visualize import np_to_image, project_root


signal_end_training = False


def translate_score(x):
    prior = 0.5
    x_logodds = math.log(x / (1-x), 2)
    prior_logodds = math.log(prior / (1-prior), 2)
    return x_logodds - prior_logodds


def report_step(step, loss, best_loss, x, x_pred):
    print('Step: {step}, Score: {score:.3f} ({raw_loss:.3f})'.format(
        step=step,
        score=-translate_score(loss),
        raw_loss=loss,
    ), end=' ')
    if loss.item() < best_loss:
        print('Model improved from {:.3f} to {:.3f}'.format(best_loss, loss.item()), end=' ')
    if step % 10 == 0:
        print('Saving images', end=' ')
        im_x = np_to_image(x[0].detach().numpy())
        im_x.save(os.path.join(project_root, 'vis', 'x.png'))
        im_pred = np_to_image(x_pred[0].detach().numpy())
        im_pred.save(os.path.join(project_root, 'vis', 'pred.png'))
    print('')


# Anton:
# If I cut down the encoding layer it seems to not be able to get to color
# It I make it shallow enough to retain color then it's basically an identity fn
# Next: Maybe there's a way to get HSV to work?
# Next: go deeeeeeep, give the encoding layer a big dimension but softmax it
# or possibly just read a bunch of papers on how to get autoencoders to derive deep properties


def main():
    batch_size = 64
    num_steps = 300

    run_timestamp = datetime.now()

    model = FaceTranslationModel()
    batch_stream = DataGen('anime').batch_stream(batch_size)

    loss_fn = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')

    for step in range(num_steps):
        x = torch.tensor(next(batch_stream)).permute(0, 3, 1, 2).float()
        x_pred = model(x)
        loss = loss_fn(x, x_pred)

        report_step(step, loss.item(), best_loss, x, x_pred)
        if loss.item() < best_loss:
            save_model(model, run_timestamp)
            best_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if signal_end_training:
            break


def handle_signal(signum, stack_frame):
    global signal_end_training
    signal_end_training = True


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, handle_signal)
    torch.set_num_threads(4)
    main()
