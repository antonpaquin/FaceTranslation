import os

import numpy as np
from PIL import Image

from FaceTranslation.util import project_root


output_dim = 256


def pad_to_square(im: Image) -> Image:
    max_dim = max(im.size)
    new_im = Image.new("RGB", size=(max_dim, max_dim), color=(0, 0, 0))
    new_im.paste(im, ((max_dim - im.size[0]) // 2, (max_dim - im.size[1]) // 2))
    return new_im


def clean_image(im: Image) -> np.ndarray:
    im = pad_to_square(im)
    im = im.resize((output_dim, output_dim), Image.BILINEAR)
    arr = np.array(im.getdata()).reshape((output_dim, output_dim, 3))
    return arr / 255


def clean_set(set_name: str) -> None:
    raw_data_path = os.path.join(project_root, 'data', 'raw')
    clean_data_path = os.path.join(project_root, 'data', 'cleaned')
    os.makedirs(os.path.join(clean_data_path, set_name), exist_ok=True)

    for idx, image_name in enumerate(os.listdir(os.path.join(raw_data_path, set_name))):
        raw_filename = os.path.join(raw_data_path, set_name, image_name)
        clean_filename = os.path.join(clean_data_path, set_name, '{:05d}.npy'.format(idx))

        im = Image.open(raw_filename)
        arr = clean_image(im)
        np.save(clean_filename, arr)


def main():
    clean_set('anime')
    clean_set('human')


if __name__ == '__main__':
    main()
