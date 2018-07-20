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
    template_pack, answer_packs = \
        tx.utils.prepare_template(data_batch, args.mask_num,
                                  args.min_mask_length, mask_id)

    # Model architecture
    embedder = tx.modules.WordEmbedder(vocab_size=train_data.vocab.size,
                                       hparams=args.word_embedding_hparams)
    encoder = tx.modules.UnidirectionalRNNEncoder(hparams=encoder_hparams)
    decoder = tx.modules.BasicRNNDecoder(vocab_size=train_data.vocab.size,
                                         hparams=decoder_hparams)
    decoder_initial_state_size = decoder.cell.state_size
    connector = tx.modules.connectors.ForwardConnector(decoder_initial_state_size)

    enc_input_embedded = embedder(template_pack['text_ids'])  # template
    dec_input_embedded = embedder(data_batch['text_ids'][:, :-1])

    _, ecdr_states = encoder(
        enc_input_embedded,
        sequence_length=data_batch["length"])

    dcdr_states = connector(ecdr_states)

    outputs, _, _ = decoder(
        initial_state=dcdr_states,
        decoding_strategy="train_greedy",
        inputs=dec_input_embedded,
        sequence_length=data_batch["length"]-1)

    mle_loss = tx.utils.smoothing_cross_entropy(
        outputs.logits,
        data_batch['text_ids'][:, 1:],
        train_data.vocab.size,
        loss_hparams['label_confidence'],
    )
    mle_loss = \
        tf.reduce_sum(mle_loss * tf.cast(template_pack['masks'][:, 1:], tf.float32)) / \
        tf.cast(tf.reduce_sum(template_pack['masks'][:, 1:]), tf.float32)

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

    # # ---unconditional---
    # all_masked = tf.fill(tf.shape(masked_inputs), mask_id)
    # all_masked_embed = embedder(all_masked)
    #
    # _, ecdr_states_uncond = encoder(
    #     all_masked_embed,
    #     sequence_length=data_batch["length"])
    #
    # dcdr_states_uncond = connector(ecdr_states_uncond)
    #
    # bos_id = train_data.vocab.token_to_id_map_py[SpecialTokens.BOS]
    # eos_id = train_data.vocab.token_to_id_map_py[SpecialTokens.EOS]
    # outputs_infer, _, _ = decoder(
    #     decoding_strategy="infer_sample",
    #     start_tokens=tf.cast(tf.fill(tf.shape(all_masked), bos_id)[:, 0], tf.int32),
    #     end_token=eos_id,
    #     embedding=embedder,
    #     initial_state=dcdr_states_uncond)

    eval_saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    def _train_epochs(session, cur_epoch):
        iterator.switch_to_train_data(session)
        while True:
            try:
                fetches = {'template': template_pack,
                           'holes': answer_packs,
                           'train_op': train_op,
                           'step': global_step,
                           'loss': mle_loss}
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.TRAIN}
                rtns = session.run(fetches, feed_dict=feed)
                step, template_, holes_, loss = rtns['step'], \
                    rtns['template'], rtns['holes'], rtns['loss']
                if step % 500 == 0:
                    rst = 'step:%s source:%s loss:%s' % \
                          (step, template_['text_ids'].shape, loss)
                    print(rst)
                if step == opt_hparams['max_training_steps']:
                    print('reach max steps:{} loss:{}'.format(step, loss))
                    print('reached max training steps')
                    return 'finished'
            except tf.errors.OutOfRangeError:
                break
        return 'done'

    def _test_epoch(cur_sess, cur_epoch):
        def _id2word_map(id_arrays):
            return [' '.join([train_data.vocab._id_to_token_map_py[i]
                              for i in sent]) for sent in id_arrays]

        iterator.switch_to_test_data(cur_sess)
        targets_list, hypothesis_list = [], []
        while True:
            try:
                fetches = {
                    'data_batch': data_batch,
                    'predictions': predictions,
                    'template': template_pack,
                    'step': global_step,
                }
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.EVAL}
                rtns = cur_sess.run(fetches, feed_dict=feed)
                templates_, targets_, predictions_ = rtns['template']['text_ids'], \
                                                    rtns['data_batch']['text_ids'],\
                                                    rtns['predictions']
                filled_templates = tx.utils.fill_template(templates_, predictions_, mask_id)

                targets, generateds = _id2word_map(targets_), _id2word_map(filled_templates)
                for target, generated in zip(targets, generateds):
                    target = target.split('<EOS>')[0].strip().split()
                    got = generated.split('<EOS>')[0].strip().split()
                    targets_list.append(target)
                    hypothesis_list.append(got)
            except tf.errors.OutOfRangeError:
                break

        outputs_tmp_filename = args.log_dir + \
            'my_model_epoch{}.beam{}alpha{}.outputs.tmp'.\
                format(cur_epoch, args.beam_width, args.alpha)
        refer_tmp_filename = os.path.join(args.log_dir, 'eval_reference.tmp')
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
             codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
            for hyp, tgt in zip(hypothesis_list, targets_list):
                tmpfile.write(' '.join(hyp) + '\n')
                tmpreffile.write(' '.join(tgt) + '\n')
        eval_bleu = float(100 * bleu_tool.bleu_wrapper(\
            refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
        print('epoch:{} eval_bleu:{}'.format(cur_epoch, eval_bleu))
        if args.save_eval_output:
            output_filename = \
                args.log_dir + 'my_model_epoch{}.beam{}alpha{}.outputs.bleu{:.3f}'\
                    .format(cur_epoch, args.beam_width, args.alpha, eval_bleu)
            result_filename = \
                args.log_dir + 'my_model_epoch{}.beam{}alpha{}.results.bleu{:.3f}'\
                    .format(cur_epoch, args.beam_width, args.alpha, eval_bleu)
            with codecs.open(output_filename, 'w+', 'utf-8') as outputfile, \
                 codecs.open(result_filename, 'w+', 'utf-8') as resultfile:
                for tgt, hyp in zip(targets_list, hypothesis_list):
                    outputfile.write(' '.join(hyp) + '\n')
                    resultfile.write("- expected: " + ' '.join(tgt) + '\n')
                    resultfile.write('- got: ' + ' '.join(hyp) + '\n\n')
        return eval_bleu

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        lowest_loss, highest_bleu, best_epoch = -1, -1, -1
        if args.running_mode == 'train_and_evaluate':
            for epoch in range(args.max_train_epoch):
                status = _train_epochs(sess, epoch)
                # test_score = _test_epoch(sess, epoch)
                # if highest_bleu < 0 or test_score > highest_bleu:
                #     print('the %d epoch, highest bleu %f' % (epoch, test_score))
                #     eval_saver.save(sess, args.log_dir + 'my-model-highest_bleu.ckpt')
                #     highest_bleu = test_score
                # if status == 'finished':
                #     print('saving model for max training steps')
                #     os.makedirs(args.log_dir + '/max/')
                #     eval_saver.save(sess, \
                #                     args.log_dir + '/max/my-model-highest_bleu.ckpt')
                #     break
                sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run(main=_main)