"""
Data loader for SeqGAN.
"""

import numpy as np
import random

from utils import pad_to_length, sent_to_ids


class GenDataLoader:
    def __init__(self, config, text_file, vocab_file, epoch_num):
        self.max_len = config.num_steps
        self.batch_size = config.batch_size
        self.epoch_num = epoch_num
        self.trained_epoch = 0
        self.step = 0
        self.eos = "<EOS>"
        self.bos = "<BOS>"
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        self.ids = self.load_data(text_file, vocab_file)
        self.batches, self.batch_num = self.create_batch()
        self.batch_ptr = 0

    def load_data(self, text_file, vocab_file):
        """
        :param text_file:
        :param vocab_file:
        :return: ids: a list of int. [sent_num, max_len + 2]
        """
        with open(vocab_file, "rb") as fin:
            words = fin.readlines()
            words = [word.strip().decode('utf-8') for word in words]
        for word in [self.eos, self.bos, self.unk, self.pad]:
            if word not in words:
                words.append(word)
        self.id2word, self.word2id = {}, {}
        for word, idx in zip(words, range(len(words))):
            self.word2id[word] = idx
            self.id2word[idx] = word

        self.bos_id = self.word2id[self.bos]
        self.eos_id = self.word2id[self.eos]
        self.unk_id = self.word2id[self.unk]
        self.pad_id = self.word2id[self.pad]

        with open(text_file, "rb") as fin:
            data = fin.readlines()
        data = [pad_to_length(sent.decode('utf-8').split(), pad=self.pad, max_len=self.max_len) for sent in data]
        ids = [sent_to_ids(sent, word2id=self.word2id, unk_id=self.word2id[self.unk])
               for sent in data]

        return ids

    def create_batch(self):
        batch_num = int(len(self.ids) / self.batch_size)
        sent_ids = np.array(self.ids[:batch_num * self.batch_size], dtype=np.int32)
        batches = np.split(sent_ids, batch_num, axis=0)
        return batches, batch_num

    def get_batch(self):
        tmp_pos = self.batch_ptr
        self.batch_ptr = self.batch_ptr + 1
        self.step = self.step + 1
        if self.batch_ptr == self.batch_num:
            self.batch_ptr = 0
            self.trained_epoch += 1
        return self.batches[tmp_pos]

    def should_stop(self):
        return self.epoch_num <= self.trained_epoch

    def reset(self):
        self.trained_epoch = 0
        self.batch_ptr = 0
        self.step = 0


class DisDataLoader:
    def __init__(self, config, positive_file, negative_file, vocab_file, epoch_num):
        self.max_len = config.num_steps
        self.batch_size = config.batch_size
        self.epoch_num = epoch_num
        self.trained_epoch = 0
        self.step = 0
        self.eos = "<EOS>"
        self.bos = "<BOS>"
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        self.r_ids, self.g_ids = self.load_data(positive_file, negative_file, vocab_file)
        self.r_id_batches, self.g_id_batches, self.batch_num = self.create_batch()
        self.batch_ptr = 0

    def load_data(self, positive_file, negative_file, vocab_file):
        """
        :param positive_file:
        :param negative_file:
        :param vocab_file:
        :return: ids: [sent_num, max_len + 1] (no <BOS>)
                labels: [sent_num, 1]
        """
        with open(vocab_file, "rb") as fin:
            words = fin.readlines()
            words = [word.strip().decode('utf-8') for word in words]
        for word in [self.eos, self.bos, self.unk, self.pad]:
            if word not in words:
                words.append(word)
        self.id2word, self.word2id = {}, {}
        for word, idx in zip(words, range(len(words))):
            self.word2id[word] = idx
            self.id2word[idx] = word

        self.bos_id = self.word2id[self.bos]
        self.eos_id = self.word2id[self.eos]
        self.unk_id = self.word2id[self.unk]
        self.pad_id = self.word2id[self.pad]

        with open(positive_file, "rb") as fin:
            positive_data = fin.readlines()
        with open(negative_file, "rb") as fin:
            negative_data = fin.readlines()

        r_data = [pad_to_length(sent.decode('utf-8').split(), pad=self.pad,
                                max_len=self.max_len) for sent in positive_data]
        r_ids = [sent_to_ids(sent, word2id=self.word2id, unk_id=self.word2id[self.unk])
                 for sent in r_data]

        g_data = [pad_to_length(sent.decode('utf-8').split(), pad=self.pad,
                                max_len=self.max_len) for sent in negative_data]
        g_ids = [sent_to_ids(sent, word2id=self.word2id, unk_id=self.word2id[self.unk])
                 for sent in g_data]

        return r_ids, g_ids

    def create_batch(self):
        batch_num = int(len(self.r_ids) / self.batch_size)
        r_sent_ids = np.array(self.r_ids[:batch_num * self.batch_size], dtype=np.int32)
        r_id_batches = np.split(r_sent_ids, batch_num, axis=0)
        g_sent_ids = np.array(self.g_ids[:batch_num * self.batch_size], dtype=np.int32)
        g_id_batches = np.split(g_sent_ids, batch_num, axis=0)
        return r_id_batches, g_id_batches, batch_num

    def get_batch(self):
        tmp_pos = self.batch_ptr
        self.batch_ptr = self.batch_ptr + 1
        self.step = self.step + 1
        if self.batch_ptr == self.batch_num:
            self.batch_ptr = 0
            self.trained_epoch += 1
        return self.r_id_batches[tmp_pos], self.g_id_batches[tmp_pos]

    def should_stop(self):
        return self.epoch_num <= self.trained_epoch

    def reset(self):
        self.trained_epoch = 0
        self.batch_ptr = 0
        self.step = 0
