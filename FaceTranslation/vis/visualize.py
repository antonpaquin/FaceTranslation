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
    return Image.fromarray(arr, 'RGB')


def main():
    data_stream = DataGen('anime').stream()
    model = load_model()

    while True:
        x = torch.tensor(next(data_stream)).permute(2, 0, 1).float()
        x = x[None, :, :, :]
        x_pred = model(x)

        im_x = np_to_image(x.detach().numpy())
        im_x.save(os.path.join(project_root, 'vis', 'x.png'))

        im_pred = np_to_image(x_pred.detach().numpy())
        im_pred.save(os.path.join(project_root, 'vis', 'pred.png'))

        input('next >')


if __name__ == '__main__':
    main()
