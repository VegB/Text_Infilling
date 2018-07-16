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

from data_utils import prepare_data_batch
import hyperparams
import bleu_tool


def _main(_):
    hparams = hyperparams.load_hyperparams()
    train_dataset_hparams, valid_dataset_hparams, test_dataset_hparams, \
    encoder_hparams, decoder_hparams, opt_hparams, loss_hparams, args = \
        hparams['train_dataset_hparams'], hparams['eval_dataset_hparams'], \
        hparams['test_dataset_hparams'], \
        hparams['encoder_hparams'], hparams['decoder_hparams'], \
        hparams['opt_hparams'], hparams['loss_hparams'], hparams['args']

    # Data
    train_data = tx.data.MonoTextData(train_dataset_hparams)
    valid_data = tx.data.MonoTextData(valid_dataset_hparams)
    test_data = tx.data.MonoTextData(test_dataset_hparams)  # ['text', 'length', 'text_ids']
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=valid_data,
                                             test=test_data)
    data_batch = iterator.get_next()
    mask_id = train_data.vocab.token_to_id_map_py['<m>']
    mask, masked_inputs, labels = \
        prepare_data_batch(args, data_batch, mask_id, args.present_rate)

    enc_paddings = tf.to_float(tf.equal(masked_inputs, 0))
    dec_paddings = tf.to_float(tf.equal(labels[:, :-1], 0))
    is_target = tf.to_float(tf.not_equal(labels[:, 1:], 0))

    # Model architecture
    embedder = tx.modules.WordEmbedder(vocab_size=train_data.vocab.size,
                                       hparams=args.word_embedding_hparams)
    encoder = tx.modules.TransformerEncoder(embedding=embedder._embedding,
                                            hparams=encoder_hparams)
    decoder = tx.modules.TemplateTransformerDecoder(embedding=embedder._embedding,
                                            hparams=decoder_hparams)

    # ---conditional---
    input_embedded = embedder(masked_inputs)

    # for loop here
    logits, preds = decoder(
        labels[:, :-1],
        input_embedded,
        None,
    )
    predictions = decoder.dynamic_decode(
        input_embedded,
        None,
    )

    mle_loss = tx.utils.smoothing_cross_entropy(
        logits,
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
    all_masked_paddings = tf.to_float(tf.equal(all_masked, 0))

    encoder_outputs_uncond, encoder_decoder_attention_bias_uncond = \
        encoder(all_masked, all_masked_paddings, None)

    predictions_infer = decoder.dynamic_decode(
        encoder_outputs_uncond,
        encoder_decoder_attention_bias_uncond,
    )

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
                           'predict': preds,
                           'mask': mask,
                           'train_op': train_op,
                           'step': global_step,
                           'loss': mle_loss}
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.TRAIN}
                rtns = session.run(fetches, feed_dict=feed)
                step, source, target, loss = rtns['step'], \
                    rtns['source'], rtns['target'], \
                    rtns['loss']
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
                    'predictions': predictions_infer,
                    'source': all_masked,
                    'target': labels,
                    'step': global_step,
                    'mle_loss': mle_loss,
                }
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.EVAL}
                rtns = cur_sess.run(fetches, feed_dict=feed)
                sources, sampled_ids, targets = \
                    rtns['source'].tolist(), \
                    rtns['predictions']['sampled_ids'][:, 0, :].tolist(), \
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
            'my_model_epoch{}.beam{}alpha{}.outputs.tmp'.format(\
            cur_epoch, args.beam_width, args.alpha)
        refer_tmp_filename = os.path.join(args.log_dir, 'eval_reference.tmp')
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
            for hyp, tgt in zip(hypothesis_list, targets_list):
                tmpfile.write(' '.join(hyp) + '\n')
                tmpreffile.write(' '.join(tgt) + '\n')
        eval_bleu = float(100 * bleu_tool.bleu_wrapper(\
            refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
        eloss = float(np.average(np.array(eloss)))
        print('epoch:{} eval_bleu:{} eval_loss:{}'.format(cur_epoch, \
            eval_bleu, eloss))
        if args.save_eval_output:
            with codecs.open(args.log_dir + \
                'my_model_epoch{}.beam{}alpha{}.outputs.bleu{:.3f}'.format(\
                cur_epoch, args.beam_width, args.alpha, eval_bleu), \
                'w+', 'utf-8') as outputfile, codecs.open(args.log_dir + \
                'my_model_epoch{}.beam{}alpha{}.results.bleu{:.3f}'.format(\
                cur_epoch, args.beam_width, args.alpha, eval_bleu), \
                'w+', 'utf-8') as resultfile:
                for src, tgt, hyp in zip(sources_list, targets_list, \
                    hypothesis_list):
                    outputfile.write(' '.join(hyp) + '\n')
                    resultfile.write("- source: " + ' '.join(src) + '\n')
                    resultfile.write("- expected: " + ' '.join(tgt) + '\n')
                    resultfile.write('- got: ' + ' '.join(hyp)+ '\n\n')
        return {'loss': eloss,
                'bleu': eval_bleu
               }

    def _test_epoch(cur_sess, cur_mname):
        # pylint:disable=too-many-locals
        iterator.switch_to_test_data(sess)
        sources_list, targets_list, hypothesis_list = [], [], []
        test_loss, test_bleu = [], 0
        if args.debug:
            fetches = {
                'source': encoder_inputs,
                'target': labels,
                'encoder_padding': enc_paddings,
                'encoder_embedding': encoder._embedding,
                'encoder_attout': encoder.stack_output,
                'encoder_output': encoder_outputs,
                'decoder_embedding': decoder._embedding,
                'predictions': predictions,
            }
            feed = {tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT}
            rtns = cur_sess.run(fetches, feed_dict=feed)
            print('source:{}'.format(rtns['source']))
            print('target:{}'.format(rtns['target']))
            print('encoder_padding:{}'.format(rtns['encoder_padding']))
            print('encoder_embedding:{}'.format(rtns['encoder_embedding']))
            print('encoder_attout:{}'.format(rtns['encoder_attout']))
            print('encoder_output:{}'.format(rtns['encoder_output']))
            print('decoder_embedding:{}'.format(rtns['decoder_embedding']))
            print('predictions:{}'.format(rtns['predictions']))
            sources, sampled_ids, targets = \
                rtns['source'].tolist(), \
                rtns['predictions']['sampled_ids'][:, 0, :].tolist(), \
                rtns['target'].tolist()
            exit()

        while True:
            try:
                fetches = {
                    'predictions': predictions,
                    'source': encoder_inputs,
                    'target': labels,
                    'step': global_step,
                    'mle_loss': mle_loss,
                    'encoder_output': encoder_outputs,
                    'decoder_input': decoder_inputs,
                    'embedding': encoder._embedding,
                    'encoder_decoder_attention_bias':
                        encoder_decoder_attention_bias,
                    'logits': logits,
                }
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT}
                rtns = cur_sess.run(fetches, feed_dict=feed)
                sources, sampled_ids, targets = \
                    rtns['source'].tolist(), \
                    rtns['predictions']['sampled_ids'][:, 0, :].tolist(), \
                    rtns['target'].tolist()
                test_loss.append(rtns['mle_loss'])
                def _id2word_map(id_arrays):
                    return [' '.join([train_data.vocab._id_to_token_map_py[i] \
                            for i in sent]) for sent in id_arrays]
                if args.debug:
                    print('source_ids:%s\ntargets_ids:%s\nsampled_ids:%s', \
                        sources, targets, sampled_ids)
                    print('encoder_output:%s %s', \
                        rtns['encoder_output'].shape, \
                        rtns['encoder_output'])
                    print('logits:%s %s', rtns['logits'].shape, \
                        rtns['logits'])
                    exit()
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
        outputs_tmp_filename = args.model_dir + \
            '{}.test.beam{}alpha{}.outputs.decodes'.format(\
            cur_mname, args.beam_width, args.alpha)
        refer_tmp_filename = args.model_dir + 'test_reference.tmp'
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
            for hyp, tgt in zip(hypothesis_list, targets_list):
                tmpfile.write(' '.join(hyp) + '\n')
                tmpreffile.write(' '.join(tgt) + '\n')
        test_bleu = float(100 * bleu_tool.bleu_wrapper(refer_tmp_filename, \
            outputs_tmp_filename, case_sensitive=True))
        test_loss = float(np.sum(np.array(test_loss)))
        print('test_bleu:%s test_loss:%s' % (test_bleu, test_loss))
        return {'loss': test_loss,
                'bleu': test_bleu}

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