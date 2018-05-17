"""
Data loader for SeqGAN.
"""

import numpy as np
import random
import texar as tx

from utils import pad_to_length, sent_to_ids, reverse_dict


class GenDataLoader:
    def __init__(self, config, text_file, word2id):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps

        self.word2id = word2id
        self.id2word = reverse_dict(self.word2id)

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


class DisDataLoader:
    def __init__(self, config, positive_file, negative_file, word2id):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps

        self.word2id = word2id

        self.pos_text = tx.data.read_words(
            positive_file, newline_token="<EOS>")
        self.pos_text_id = [self.word2id[w] for w in self.pos_text if w in self.word2id]

        self.neg_text = tx.data.read_words(
            negative_file, newline_token="<EOS>")
        self.neg_text_id = [self.word2id[w] for w in self.neg_text if w in self.word2id]

        data_length = len(self.neg_text_id) if len(self.pos_text_id) > len(self.neg_text_id) else len(self.pos_text_id)
        batch_length = data_length // self.batch_size

        pos_data = np.asarray(self.pos_text_id[:self.batch_size * batch_length])
        self.pos_data = pos_data.reshape([self.batch_size, batch_length])

        neg_data = np.asarray(self.neg_text_id[:self.batch_size * batch_length])
        self.neg_data = neg_data.reshape([self.batch_size, batch_length])

        self.epoch_size = (batch_length - 1) // self.num_steps
        if self.epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    def iter(self):
        for i in range(self.epoch_size):
            pos = self.pos_data[:, i * self.num_steps: (i + 1) * self.num_steps]
            neg = self.neg_data[:, i * self.num_steps: (i + 1) * self.num_steps]
            yield (pos, neg)
