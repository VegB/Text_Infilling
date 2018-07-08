# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import os
import sys
import time
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from data_utils import prepare_data_batch

flags = tf.flags
flags.DEFINE_string("dataset", "yahoo",
                    "perform training on ptb or yahoo.")
flags.DEFINE_string("data_path", "./",
                    "Directory containing PTB or Yahoo raw data. "
                    "If not exists, the directory will be created, "
                    "and the data will be downloaded.")
flags.DEFINE_string("mask_strategy", "random",
                    "random / contiguous")
flags.DEFINE_string("config", "config", "The config to use.")
FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)


def prepare_data(train_path):
    """Download the PTB or Yahoo dataset
    """
    data_path = FLAGS.data_path

    if not tf.gfile.Exists(train_path):
        url = 'https://jxhe.github.io/download/yahoo_data.tgz'
        tx.data.maybe_download(url, data_path, extract=True)
        os.remove('%s_data.tgz' % FLAGS.dataset)

        data_path = os.path.join(data_path, '%s_data' % FLAGS.dataset)

        train_path = os.path.join(data_path, "%s.train.txt" % FLAGS.dataset)
        valid_path = os.path.join(data_path, "%s.valid.txt" % FLAGS.dataset)
        test_path = os.path.join(data_path, "%s.test.txt" % FLAGS.dataset)
        vocab_path = os.path.join(data_path, "vocab.txt")

        config.train_data_hparams['dataset'] = {'files': train_path,
                                                'vocab_file': vocab_path}

        config.val_data_hparams['dataset'] = {'files': valid_path,
                                              'vocab_file': vocab_path}

        config.test_data_hparams['dataset'] = {'files': test_path,
                                               'vocab_file': vocab_path}


def _main(_):
    prepare_data(config.train_data_hparams['dataset']['files'])

    # Data
    train_data = tx.data.MonoTextData(config.train_data_hparams)
    val_data = tx.data.MonoTextData(config.val_data_hparams)
    test_data = tx.data.MonoTextData(config.test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()
    masks, encoder_inputs, decoder_inputs, labels = \
        prepare_data_batch(config, data_batch)
    enc_paddings = tf.to_float(tf.equal(encoder_inputs, 0))
    dec_paddings = tf.to_float(tf.equal(decoder_inputs, 0))
    is_target = tf.to_float(tf.not_equal(labels, 0))

    opt_vars = {
        'learning_rate': config.lr_decay_hparams["init_lr"],
        'best_valid_nll': 1e100,
        'steps_not_improved': 0,
        'kl_weight': config.kl_anneal_hparams["start"]
    }

    # Model architecture
    embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab.size, hparams=config.emb_hparams)

    encoder = tx.modules.TransformerEncoder(embedding=embedder._embedding,
        hparams=config.encoder_hparams)
    encoder_outputs, encoder_decoder_attention_bias = \
        encoder(encoder_inputs, enc_paddings)

    decoder = tx.modules.TransformerDecoder(embedding=embedder._embedding,
        hparams=config.decoder_hparams)
    logits, preds = decoder(
        decoder_inputs,
        encoder_outputs,
        encoder_decoder_attention_bias,
    )
    predictions = decoder.dynamic_decode(
        encoder_outputs,
        encoder_decoder_attention_bias,
    )

    mle_loss = tx.utils.smoothing_cross_entropy(
        logits,
        labels,
        train_database.target_vocab.size,
        loss_hparams['label_confidence'],
    )
    mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)
