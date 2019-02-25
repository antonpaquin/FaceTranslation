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


def get_x(batch):
    return torch.tensor(next(batch)).permute(0, 3, 1, 2).float()


def main():
    human_stream = DataGen('human').batch_stream(batch_size=8)
    anime_stream = DataGen('anime').batch_stream(batch_size=8)
    model = load_model()

    cross = True
    n = 0
    while True:
        if (n % 2 == 0):
            x = get_x(anime_stream)
            key_enc = torch.zeros((8,))
        else:
            x = get_x(human_stream)
            key_enc = torch.ones((8,))

        if cross:
            key_enc = 1 - key_enc

        n += 1

        x_pred, _ = model(x, key_enc)

        print(x.size())
        print(x.min(), x.max())
        im_x = np_to_image(x.detach().numpy())
        im_x.save(os.path.join(project_root, 'vis', 'x.png'))

        print(x_pred.size())
        print((x_pred.min().item(), x_pred.max().item()))
        im_pred = np_to_image(x_pred.detach().numpy())
        im_pred.save(os.path.join(project_root, 'vis', 'pred.png'))

        input('next >')


if __name__ == '__main__':
    main()
