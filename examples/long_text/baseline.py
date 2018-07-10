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
import logging
import numpy as np
import tensorflow as tf
import texar as tx
from texar.data import SpecialTokens

from data_utils import prepare_data_batch
import baseline_hyperparams
import bleu_tool


def _main(_):
    hparams = baseline_hyperparams.load_hyperparams()
    train_dataset_hparams, valid_dataset_hparams, test_dataset_hparams, \
    encoder_hparams, decoder_hparams, opt_hparams, loss_hparams, args = \
        hparams['train_dataset_hparams'], hparams['eval_dataset_hparams'], \
        hparams['test_dataset_hparams'], \
        hparams['encoder_hparams'], hparams['decoder_hparams'], \
        hparams['opt_hparams'], hparams['loss_hparams'], hparams['args']

    # Data
    train_data = tx.data.MonoTextData(train_dataset_hparams)
    valid_data = tx.data.MonoTextData(valid_dataset_hparams)
    test_data = tx.data.MonoTextData(test_dataset_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=valid_data,
                                             test=test_data)
    data_batch = iterator.get_next()
    mask_id = train_data.vocab.token_to_id_map_py['<m>']
    mask, masked_inputs, labels = \
        prepare_data_batch(args, data_batch, mask_id, args.present_rate)
    is_target = tf.to_float(tf.not_equal(labels[:, 1:], 0))

    # Model architecture
    embedder = tx.modules.WordEmbedder(vocab_size=train_data.vocab.size,
                                       hparams=args.word_embedding_hparams)
    encoder = tx.modules.UnidirectionalRNNEncoder(hparams=encoder_hparams)
    decoder = tx.modules.BasicRNNDecoder(vocab_size=train_data.vocab.size,
                                         hparams=decoder_hparams)
    decoder_initial_state_size = decoder.cell.state_size
    connector = tx.modules.connectors.ForwardConnector(decoder_initial_state_size)

    enc_input_embed = embedder(masked_inputs)
    dec_input_embed = embedder(labels)

    _, ecdr_states = encoder(
        enc_input_embed,
        sequence_length=data_batch["length"])

    dcdr_states = connector(ecdr_states)

    outputs, _, _ = decoder(
        initial_state=dcdr_states,
        decoding_strategy="train_greedy",
        inputs=dec_input_embed,
        sequence_length=data_batch["length"]-1)

    mle_loss = tx.utils.smoothing_cross_entropy(
        outputs.logits,
        labels[:, 1:],
        train_data.vocab.size,
        loss_hparams['label_confidence'],
    )
    mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

    global_step = tf.Variable(0, trainable=False)
    fstep = tf.to_float(global_step)
    if opt_hparams['learning_rate_schedule'] == 'static':
        learning_rate = 1e-3
    else:
        learning_rate = opt_hparams['lr_constant'] \
                        * tf.minimum(1.0, (fstep / opt_hparams['warmup_steps'])) \
                        * tf.rsqrt(tf.maximum(fstep, opt_hparams['warmup_steps'])) \
                        * args.hidden_dim ** -0.5
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=opt_hparams['Adam_beta1'],
        beta2=opt_hparams['Adam_beta2'],
        epsilon=opt_hparams['Adam_epsilon'],
    )
    train_op = optimizer.minimize(mle_loss, global_step)

    # ---unconditional---
    all_masked = tf.fill(tf.shape(masked_inputs), mask_id)
    all_masked_embed = embedder(all_masked)

    _, ecdr_states_uncond = encoder(
        all_masked_embed,
        sequence_length=data_batch["length"])

    dcdr_states_uncond = connector(ecdr_states_uncond)

    bos_id = train_data.vocab.token_to_id_map_py[SpecialTokens.BOS]
    eos_id = train_data.vocab.token_to_id_map_py[SpecialTokens.EOS]
    outputs_infer, _, _ = decoder(
        decoding_strategy="infer_sample",
        start_tokens=tf.cast(tf.fill(tf.shape(all_masked), bos_id)[:, 0], tf.int32),
        end_token=eos_id,
        embedding=embedder,
        initial_state=dcdr_states_uncond)

    eval_saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    def _train_epochs(session, cur_epoch):
        iterator.switch_to_train_data(session)
        while True:
            try:
                fetches = {'source': masked_inputs,
                           'dec_in': labels[:, :-1],
                           'target': labels[:, 1:],
                           'mask': mask,
                           'train_op': train_op,
                           'step': global_step,
                           'loss': mle_loss}
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.TRAIN}
                rtns = session.run(fetches, feed_dict=feed)
                step, source, target, loss = rtns['step'], rtns['source'], \
                                             rtns['target'], rtns['loss']
                if step % 100 == 0:
                    rst = 'step:%s source:%s targets:%s loss:%s' % \
                          (step, source.shape, target.shape, loss)
                    print(rst)
                    # print(rtns['mask'])
                if step == opt_hparams['max_training_steps']:
                    print('reach max steps:{} loss:{}'.format(step, loss))
                    print('reached max training steps')
                    return 'finished'
            except tf.errors.OutOfRangeError:
                break
        return 'done'

    def _eval_epoch(cur_sess, cur_epoch):
        # pylint:disable=too-many-locals
        iterator.switch_to_val_data(cur_sess)
        sources_list, targets_list, hypothesis_list = [], [], []
        eloss = []
        while True:
            try:
                fetches = {
                    'outputs': outputs_infer,
                    'source': all_masked,
                    'target': labels,
                    'step': global_step,
                    'mle_loss': mle_loss,
                }
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.EVAL}
                rtns = cur_sess.run(fetches, feed_dict=feed)
                sources, sampled_ids, targets = \
                    rtns['source'].tolist(), \
                    rtns['outputs'].sample_id[:, 0, :].tolist(), \
                    rtns['target'].tolist()
                eloss.append(rtns['mle_loss'])
                if args.verbose:
                    print('cur loss:{}'.format(rtns['mle_loss']))

                def _id2word_map(id_arrays):
                    return [' '.join([train_data.vocab._id_to_token_map_py[i]
                                      for i in sent]) for sent in id_arrays]

                sources, targets, dwords = _id2word_map(sources), \
                                           _id2word_map(targets), \
                                           _id2word_map(sampled_ids)
                for source, target, pred in zip(sources, targets, dwords):
                    source = source.split('<EOS>')[0].strip().split()
                    target = target.split('<EOS>')[0].strip().split()
                    got = pred.split('<EOS>')[0].strip().split()
                    sources_list.append(source)
                    targets_list.append(target)
                    hypothesis_list.append(got)
            except tf.errors.OutOfRangeError:
                break
        outputs_tmp_filename = args.log_dir + \
                               'my_model_epoch{}.beam{}alpha{}.outputs.tmp'.format( \
                                   cur_epoch, args.beam_width, args.alpha)
        refer_tmp_filename = os.path.join(args.log_dir, 'eval_reference.tmp')
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
            for hyp, tgt in zip(hypothesis_list, targets_list):
                tmpfile.write(' '.join(hyp) + '\n')
                tmpreffile.write(' '.join(tgt) + '\n')
        eval_bleu = float(100 * bleu_tool.bleu_wrapper( \
            refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
        eloss = float(np.average(np.array(eloss)))
        print('epoch:{} eval_bleu:{} eval_loss:{}'.format(cur_epoch, \
                                                          eval_bleu, eloss))
        if args.save_eval_output:
            with codecs.open(args.log_dir + \
                                     'my_model_epoch{}.beam{}alpha{}.outputs.bleu{:.3f}'.format( \
                                             cur_epoch, args.beam_width, args.alpha, eval_bleu), \
                             'w+', 'utf-8') as outputfile, codecs.open(args.log_dir + \
                                                                               'my_model_epoch{}.beam{}alpha{}.results.bleu{:.3f}'.format( \
                                                                                       cur_epoch, args.beam_width,
                                                                                   args.alpha, eval_bleu), \
                                                                       'w+', 'utf-8') as resultfile:
                for src, tgt, hyp in zip(sources_list, targets_list, \
                                         hypothesis_list):
                    outputfile.write(' '.join(hyp) + '\n')
                    resultfile.write("- source: " + ' '.join(src) + '\n')
                    resultfile.write("- expected: " + ' '.join(tgt) + '\n')
                    resultfile.write('- got: ' + ' '.join(hyp) + '\n\n')
        return {'loss': eloss,
                'bleu': eval_bleu
                }

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        var_list = tf.trainable_variables()
        with open(args.log_dir + 'var.list', 'w+') as outfile:
            for var in var_list:
                outfile.write('var:{} shape:{} dtype:{}\n'.format( \
                    var.name, var.shape, var.dtype))
        lowest_loss, highest_bleu, best_epoch = -1, -1, -1
        if args.running_mode == 'train_and_evaluate':
            for epoch in range(args.max_train_epoch):
                if epoch % args.eval_interval_epoch != 0:
                    continue
                status = _train_epochs(sess, epoch)
                eval_result = _eval_epoch(sess, epoch)
                eval_loss, eval_score = eval_result['loss'], eval_result['bleu']
                if args.eval_criteria == 'loss':
                    if lowest_loss < 0 or eval_loss < lowest_loss:
                        print('the %s epoch got lowest loss %s', \
                              epoch, eval_loss)
                        eval_saver.save(sess, \
                                        args.log_dir + 'my-model-lowest_loss.ckpt')
                        lowest_loss = eval_loss
                elif args.eval_criteria == 'bleu':
                    if highest_bleu < 0 or eval_score > highest_bleu:
                        print('the %s epoch, highest bleu %s', \
                              epoch, eval_score)
                        eval_saver.save(sess, \
                                        args.log_dir + 'my-model-highest_bleu.ckpt')
                        highest_bleu = eval_score
                if status == 'finished':
                    print('saving model for max training steps')
                    os.makedirs(args.log_dir + '/max/')
                    eval_saver.save(sess, \
                                    args.log_dir + '/max/my-model-highest_bleu.ckpt')
                    break
                sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run(main=_main)