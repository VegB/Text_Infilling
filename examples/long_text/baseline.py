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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import sys
import time
import codecs
import logging
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
import texar as tx
from texar.data import SpecialTokens
from texar.modules.embedders import position_embedders
from texar.utils import utils

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
    boa_id = train_data.vocab.token_to_id_map_py['<BOA>']
    eoa_id = train_data.vocab.token_to_id_map_py['<EOA>']
    bos_id = train_data.vocab.token_to_id_map_py[SpecialTokens.BOS]
    eos_id = train_data.vocab.token_to_id_map_py[SpecialTokens.EOS]
    pad_id = train_data.vocab.token_to_id_map_py['<PAD>']
    template_pack, answer_packs = \
        tx.utils.prepare_template(data_batch, args, mask_id, boa_id, eoa_id, pad_id)

    # Model architecture
    embedder = tx.modules.WordEmbedder(vocab_size=train_data.vocab.size,
                                       hparams=args.word_embedding_hparams)
    position_embedder = position_embedders.SinusoidsSegmentalPositionEmbedder()
    encoder = tx.modules.UnidirectionalRNNEncoder(hparams=encoder_hparams)
    decoder = tx.modules.BasicPositionalRNNDecoder(vocab_size=train_data.vocab.size,
                                                   hparams=decoder_hparams,
                                                   position_embedder=position_embedder)
    decoder_initial_state_size = decoder.cell.state_size
    connector = tx.modules.connectors.ForwardConnector(decoder_initial_state_size)

    template = template_pack['templates']
    template_word_embeds = embedder(template)
    template_length = utils.shape_list(template)[1]
    channels = utils.shape_list(template)[2]
    template_pos_embeds = position_embedder(template_length, channels,
                                            template_pack['segment_ids'],
                                            template_pack['offsets'])
    enc_input_embedded = template_word_embeds + template_pos_embeds

    _, ecdr_states = encoder(
        enc_input_embedded,
        sequence_length=data_batch["length"])

    dcdr_init_states = connector(ecdr_states)

    cetp_loss = None
    for hole in answer_packs:
        dec_input = hole['text_ids'][:-1]
        dec_input_word_embeds = embedder(dec_input)
        dec_input_length = utils.shape_list(dec_input)[1]
        dec_input_pos_embeds = position_embedder(dec_input_length, channels,
                                                 hole['segment_ids'],
                                                 hole['offsets'])
        dec_input_embedded = dec_input_word_embeds + dec_input_pos_embeds
        outputs, _, _ = decoder(
            initial_state=dcdr_init_states,
            decoding_strategy="train_greedy",
            inputs=dec_input_embedded,
            sequence_length=data_batch["length"]-1)

        cur_loss = tx.utils.smoothing_cross_entropy(
            outputs.logits,
            hole['text_ids'][:, 1:],
            train_data.vocab.size,
            loss_hparams['label_confidence'],
        )
        cetp_loss = cur_loss if cetp_loss is None \
            else tf.concat([cetp_loss, cur_loss], -1)

    cetp_loss = tf.reduce_mean(cetp_loss)

    global_step = tf.Variable(0, trainable=False)
    fstep = tf.to_float(global_step)
    if opt_hparams['learning_rate_schedule'] == 'static':
        learning_rate = 1e-3
    else:
        learning_rate = opt_hparams['lr_constant'] \
                        * tf.minimum(1.0, (fstep / opt_hparams['warmup_steps'])) \
                        * tf.rsqrt(tf.maximum(fstep, opt_hparams['warmup_steps'])) \
                        * args.hidden_dim ** -0.5 \
                        * args.present_rate
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=opt_hparams['Adam_beta1'],
        beta2=opt_hparams['Adam_beta2'],
        epsilon=opt_hparams['Adam_epsilon'],
    )
    train_op = optimizer.minimize(cetp_loss, global_step)

    outputs_infer, _, _ = decoder(
        decoding_strategy="infer_sample",
        start_tokens=tf.cast(tf.fill([tf.shape(data_batch['text_ids'])[0]], bos_id), tf.int32),
        end_token=eos_id,
        embedding=embedder,
        initial_state=dcdr_init_states)

    eval_saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    def _train_epochs(session, cur_epoch):
        iterator.switch_to_train_data(session)
        if args.draw_for_debug:
            loss_lists = []
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
                if args.draw_for_debug:
                    loss_lists.append(loss)
                if step == opt_hparams['max_training_steps']:
                    print('reach max steps:{} loss:{}'.format(step, loss))
                    print('reached max training steps')
                    return 'finished'
            except tf.errors.OutOfRangeError:
                break
            if args.draw_for_debug:
                plt.figure(figsize=(14, 10))
                plt.plot(loss_lists, '--', linewidth=1, label='loss trend')
                plt.ylabel('training loss')
                plt.xlabel('training steps in one epoch')
                plt.savefig(args.log_dir + '/img/train_loss_curve.epoch{}.png'.format(cur_epoch))
        return 'done'

    def _test_epoch(cur_sess, cur_epoch):
        def _id2word_map(id_arrays):
            return [' '.join([train_data.vocab._id_to_token_map_py[i]
                              for i in sent]) for sent in id_arrays]

        iterator.switch_to_test_data(cur_sess)
        templates_list, targets_list, hypothesis_list = [], [], []
        while True:
            try:
                fetches = {
                    'data_batch': data_batch,
                    'predictions': outputs_infer,
                    'template': template_pack,
                    'step': global_step,
                }
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.EVAL}
                rtns = cur_sess.run(fetches, feed_dict=feed)
                templates_, targets_, predictions_ = rtns['template']['text_ids'].tolist(), \
                                                    rtns['data_batch']['text_ids'].tolist(),\
                                                    rtns['predictions'].sample_id.tolist()

                templates, targets, generateds = \
                    _id2word_map(templates_), _id2word_map(targets_), _id2word_map(predictions_)
                for template, target, generated in zip(templates, targets, generateds):
                    template = template.split('<EOS>')[0].strip().split()
                    target = target.split('<EOS>')[0].strip().split()
                    got = generated.split('<EOS>')[0].strip().split()
                    templates_list.append(template)
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
        os.remove(outputs_tmp_filename)
        os.remove(refer_tmp_filename)
        if args.save_eval_output:
            result_filename = \
                args.log_dir + 'my_model_epoch{}.beam{}alpha{}.results.bleu{:.3f}'\
                    .format(cur_epoch, args.beam_width, args.alpha, eval_bleu)
            with codecs.open(result_filename, 'w+', 'utf-8') as resultfile:
                for tmplt, tgt, hyp in zip(templates_list, targets_list, hypothesis_list):
                    resultfile.write("- template: " + ' '.join(tmplt) + '\n')
                    resultfile.write("- expected: " + ' '.join(tgt) + '\n')
                    resultfile.write('- got: ' + ' '.join(hyp) + '\n\n')
        return eval_bleu
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        lowest_loss, highest_bleu, best_epoch = -1, -1, -1
        if args.running_mode == 'train_and_evaluate':
            for epoch in range(args.max_train_epoch):
                status = _train_epochs(sess, epoch)
                test_score = _test_epoch(sess, epoch)
                if highest_bleu < 0 or test_score > highest_bleu:
                    print('the %d epoch, highest bleu %f' % (epoch, test_score))
                    eval_saver.save(sess, args.log_dir + 'my-model-highest_bleu.ckpt')
                    highest_bleu = test_score
                if status == 'finished':
                    print('saving model for max training steps')
                    os.makedirs(args.log_dir + '/max/')
                    eval_saver.save(sess, \
                                    args.log_dir + '/max/my-model-highest_bleu.ckpt')
                    break
                sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run(main=_main)