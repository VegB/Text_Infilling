"""
Data loader for SeqGAN.
"""

import numpy as np
import texar as tx


class DataLoader:
    def __init__(self, config, text_file, word2id):
        self.batch_size = config.training_hparams['batch_size']
        self.num_steps = config.training_hparams['num_steps']

        self.word2id = word2id

        self.text = tx.data.read_words(
            text_file, newline_token="<EOS>")
        self.text_id = [self.word2id[w] for w in self.text if w in self.word2id]

        data_length = len(self.text_id)
        batch_length = data_length // self.batch_size

        data = np.asarray(self.text_id[:self.batch_size * batch_length])
        self.data = data.reshape([self.batch_size, batch_length])

        self.epoch_size = (batch_length - 1) // self.num_steps
        if self.epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    def iter(self):
        for i in range(self.epoch_size):
            x = self.data[:, i * self.num_steps: (i + 1) * self.num_steps]
            y = self.data[:, i * self.num_steps + 1: (i + 1) * self.num_steps + 1]
            yield (x, y)
