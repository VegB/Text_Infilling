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
import codecs
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from data_utils import prepare_data_batch
import bleu_tool

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

        # add mask token <m>
        with open(vocab_path, 'ab+') as fin:
            fin.write('<m>\n'.encode('utf-8'))

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
        prepare_data_batch(config, data_batch,
                           train_data.vocab.token_to_id_map_py['<m>'])

    enc_paddings = tf.to_float(tf.equal(encoder_inputs, 0))
    dec_paddings = tf.to_float(tf.equal(decoder_inputs, 0))
    is_target = tf.to_float(tf.not_equal(labels, 0))

    # Model architecture
    embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab.size, hparams=config.emb_hparams)

    encoder = tx.modules.TransformerEncoder(embedding=embedder._embedding,
        hparams=config.encoder_hparams)
    encoder_outputs, encoder_decoder_attention_bias = \
        encoder(encoder_inputs, enc_paddings, None)

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
        train_data.vocab.size,
        config.loss_hparams['label_confidence'],
    )
    mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

    global_step = tf.Variable(0, trainable=False)
    fstep = tf.to_float(global_step)
    if config.opt_hparams['learning_rate_schedule'] == 'static':
        learning_rate = 1e-3
    else:
        learning_rate = config.opt_hparams['lr_constant'] \
            * tf.minimum(1.0, (fstep / config.opt_hparams['warmup_steps'])) \
            * tf.rsqrt(tf.maximum(fstep, config.opt_hparams['warmup_steps'])) \
            * config.hidden_size**-0.5
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=config.opt_hparams['Adam_beta1'],
        beta2=config.opt_hparams['Adam_beta2'],
        epsilon=config.opt_hparams['Adam_epsilon'],
    )
    train_op = optimizer.minimize(mle_loss, global_step)

    def _run_epoch(sess, mode):
        if mode is 'train':
            iterator.switch_to_train_data(sess)
        elif mode is 'valid':
            iterator.switch_to_val_data(sess)
        elif mode is 'test':
            iterator.switch_to_test_data(sess)
        else:
            sys.exit("INVALID MODE %s, expecting one of "
                     "['train', 'valid', 'test']" % mode)

        sources_list, targets_list, hypothesis_list = [], [], []
        eloss = []

        fetches = {
            'mle_loss': mle_loss,
            'step': global_step,
            'predictions': predictions
        }
        if mode is 'train':
            fetches['train_op'] = train_op

        while True:
            try:
                feed_dict = {
                    tx.global_mode(): tf.estimator.ModeKeys.TRAIN if mode is 'train'
                    else tf.estimator.ModeKeys.EVAL
                }
                rtns = sess.run(fetches, feed_dict)
                if mode is 'train' and rtns['step'] % 100 == 0:
                    rst = "step: %d, train_ppl: %.6f" % (rtns['step'], rtns['mle_loss'])
                    print(rst)
                elif mode is 'test':
                    def _id2word_map(id_arrays):
                        return [' '.join([train_data.vocab.id_to_token_map_py[i] \
                                          for i in sent]) for sent in id_arrays]

                    sources, targets, dwords = _id2word_map(decoder_inputs), \
                                               _id2word_map(labels), \
                                               _id2word_map(predictions['sampled_ids'])
                    for source, target, pred in zip(sources, targets, dwords):
                        source = source.split('<EOS>')[0].strip().split()
                        target = target.split('<EOS>')[0].strip().split()
                        got = pred.split('<EOS>')[0].strip().split()
                        sources_list.append(source)
                        targets_list.append(target)
                        hypothesis_list.append(got)
                        eloss.append(rtns['mle_loss'])
            except tf.errors.OutOfRangeError:
                break

        if mode is 'test':
            outputs_tmp_filename = config.log_dir + \
                                   'my_model_epoch{}.beam{}alpha{}.outputs.tmp'.format(
                                   rtns['step'], config.beam_width, config.alpha)
            refer_tmp_filename = os.path.join(config.log_dir, 'eval_reference.tmp')
            with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                    codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
                for hyp, tgt in zip(hypothesis_list, targets_list):
                    tmpfile.write(' '.join(hyp) + '\n')
                    tmpreffile.write(' '.join(tgt) + '\n')
            eval_bleu = float(100 * bleu_tool.bleu_wrapper(\
                refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
            eloss = float(np.average(np.array(eloss)))
            print('epoch:{} eval_bleu:{} eval_loss:{}'.format(rtns['step'],
                                                              eval_bleu, eloss))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver()

        for g_epoch in range(config.num_epochs):
            _run_epoch(sess, 'train')
            if g_epoch % 20 == 0:
                _run_epoch(sess, 'test')
                # saver.save(sess, config.ckpt, global_step=g_epoch + 1)


if __name__ == '__main__':
    tf.app.run(main=_main)