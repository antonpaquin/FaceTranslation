import os

import torch
from PIL import Image

from FaceTranslation.model import load_model
from FaceTranslation.data.data_generator import DataGen
from FaceTranslation.util import project_root


def np_to_image(arr):
    arr = arr[0] * 255
    arr = arr.transpose(1, 2, 0)
    arr = arr.clip(0, 255)
    arr = arr.astype('uint8')
    arr = arr[:, :, 0]
    return Image.fromarray(arr, 'L')


def main():
    human_stream = DataGen('human').stream()
    anime_stream = DataGen('anime').stream()
    model = load_model()

    n = 0
    while True:
        if (n % 2 == 0):
            x = torch.tensor(next(anime_stream)).permute(2, 0, 1).float()
            x = x[None, :, :, :]
            key_enc = torch.ones_like(x)
        else:
            x = torch.tensor(next(human_stream)).permute(2, 0, 1).float()
            x = x[None, :, :, :]
            key_enc = torch.zeros_like(x)
        n += 1

        x_pred, _ = model(x, key_enc)

        print(x.size())
        im_x = np_to_image(x.detach().numpy())
        im_x.save(os.path.join(project_root, 'vis', 'x.png'))

        print(x_pred.size())
        im_pred = np_to_image(x_pred.detach().numpy())
        im_pred.save(os.path.join(project_root, 'vis', 'pred.png'))

        input('next >')


if __name__ == '__main__':
    main()
