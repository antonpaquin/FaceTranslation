import os
from random import shuffle

import numpy as np

from FaceTranslation.util import project_root


class DataGen:
    def __init__(self, dataset):
        data_root = os.path.join(project_root, 'data', 'cleaned', dataset)
        self.filenames = [os.path.join(data_root, fname) for fname in os.listdir(data_root)]

    def stream(self):
        while True:
            shuffle(self.filenames)
            for file in self.filenames:
                yield np.load(file)

    def batch_stream(self, batch_size):
        file_stream = self.stream()
        while True:
            batch = []
            for _ in range(batch_size):
                batch.append(next(file_stream))
            yield np.array(batch)
