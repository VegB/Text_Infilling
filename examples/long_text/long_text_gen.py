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
import codecs
import numpy as np
import tensorflow as tf
import texar as tx
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import hyperparams
import bleu_tool


def _main(_):
    hparams = hyperparams.load_hyperparams()
    train_dataset_hparams, valid_dataset_hparams, test_dataset_hparams, \
    decoder_hparams, opt_hparams, loss_hparams, args = \
        hparams['train_dataset_hparams'], hparams['eval_dataset_hparams'], \
        hparams['test_dataset_hparams'], hparams['decoder_hparams'], \
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
    eos_id = train_data.vocab.token_to_id_map_py['<EOS>']
    pad_id = train_data.vocab.token_to_id_map_py['<PAD>']
    template_pack, answer_packs = \
        tx.utils.prepare_template(data_batch, args, mask_id, boa_id, eoa_id, pad_id)

    # Model architecture
    embedder = tx.modules.WordEmbedder(vocab_size=train_data.vocab.size,
                                       hparams=args.word_embedding_hparams)
    decoder = \
        tx.modules.TemplateTransformerDecoder(embedding=embedder._embedding,
                                              hparams=decoder_hparams)

    cetp_loss = None
    for hole in answer_packs:
        logits, preds = decoder(decoder_input_pack=hole,
                                template_input_pack=template_pack,
                                encoder_decoder_attention_bias=None,
                                args=args)

        cur_loss = tx.utils.smoothing_cross_entropy(
            logits,
            hole['text_ids'][:, 1:],
            train_data.vocab.size,
            loss_hparams['label_confidence'])
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
                        * args.hidden_dim ** -0.5
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=opt_hparams['Adam_beta1'],
                                       beta2=opt_hparams['Adam_beta2'],
                                       epsilon=opt_hparams['Adam_epsilon'])
    train_op = optimizer.minimize(cetp_loss, global_step)

    predictions = []
    offsets = tx.utils.generate_prediction_offsets(data_batch['text_ids'],
                                                   args.max_decode_len + 1)
    for idx, _ in enumerate(answer_packs):
        segment_ids = \
            tx.utils.generate_prediction_segment_ids(data_batch['text_ids'],
                                                 idx * 2 + 1,  # segment id starting from 1
                                                 args.max_decode_len + 1)
        preds = decoder.dynamic_decode(
            template_input_pack=template_pack,
            encoder_decoder_attention_bias=None,
            segment_ids=segment_ids,
            offsets=offsets,
            bos_id=boa_id,
            eos_id=eoa_id)
        predictions.append(preds['sampled_ids'][0])

    eval_saver = tf.train.Saver(max_to_keep=5)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    def _train_epochs(session, cur_epoch):
        iterator.switch_to_train_data(session)
        loss_lists = []
        while True:
            try:
                fetches = {'template': template_pack,
                           'holes': answer_packs,
                           'train_op': train_op,
                           'step': global_step,
                           'loss': cetp_loss}
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.TRAIN}
                rtns = session.run(fetches, feed_dict=feed)
                step, template_, holes_, loss = rtns['step'], \
                    rtns['template'], rtns['holes'], rtns['loss']
                if step % 200 == 1:
                    rst = 'step:%s source:%s loss:%s' % \
                          (step, template_['text_ids'].shape, loss)
                    print(rst)
                loss_lists.append(loss)
            except tf.errors.OutOfRangeError:
                break
        return loss_lists[::50]

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
                    'predictions': predictions,
                    'template': template_pack,
                    'step': global_step,
                }
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.EVAL}
                rtns = cur_sess.run(fetches, feed_dict=feed)
                real_templates_, templates_, targets_, predictions_ = \
                    rtns['template']['templates'], rtns['template']['text_ids'], \
                                                    rtns['data_batch']['text_ids'],\
                                                    rtns['predictions']

                filled_templates = \
                    tx.utils.fill_template(templates_, predictions_, mask_id, eoa_id, pad_id, eos_id)

                templates, targets, generateds = _id2word_map(real_templates_.tolist()), \
                                                 _id2word_map(targets_), \
                                                 _id2word_map(filled_templates)

                for template, target, generated in zip(templates, targets, generateds):
                    template = template.split('<EOS>')[0].strip().split()
                    target = target.split('<EOS>')[0].strip().split()
                    got = generated.split('<EOS>')[0].strip().split()
                    templates_list.append(template)
                    targets_list.append(target)
                    hypothesis_list.append(got)
            except tf.errors.OutOfRangeError:
                break

        outputs_tmp_filename = args.log_dir + 'my_model_epoch{}.beam{}alpha{}.outputs.tmp'.\
                format(cur_epoch, args.beam_width, args.alpha)
        template_tmp_filename = args.log_dir + 'my_model_epoch{}.beam{}alpha{}.templates.tmp'.\
                format(cur_epoch, args.beam_width, args.alpha)
        refer_tmp_filename = os.path.join(args.log_dir, 'eval_reference.tmp')
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
            codecs.open(template_tmp_filename, 'w+', 'utf-8') as tmptpltfile, \
            codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
            for hyp, tplt, tgt in zip(hypothesis_list, templates_list, targets_list):
                tmpfile.write(' '.join(hyp) + '\n')
                tmptpltfile.write(' '.join(tplt) + '\n')
                tmpreffile.write(' '.join(tgt) + '\n')
        eval_bleu = float(100 * bleu_tool.bleu_wrapper(\
            refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
        template_bleu = float(100 * bleu_tool.bleu_wrapper(\
            refer_tmp_filename, template_tmp_filename, case_sensitive=True))
        print('epoch:{} eval_bleu:{} template_bleu:{}'.format(cur_epoch, eval_bleu, template_bleu))
        os.remove(outputs_tmp_filename)
        os.remove(template_tmp_filename)
        os.remove(refer_tmp_filename)
        if args.save_eval_output:
            result_filename = \
                args.log_dir + 'my_model_epoch{}.beam{}alpha{}.results.bleu{:.3f}'\
                    .format(cur_epoch, args.beam_width, args.alpha, eval_bleu)
            with codecs.open(result_filename, 'w+', 'utf-8') as resultfile:
                for tmplt, tgt, hyp in zip(templates_list, targets_list, hypothesis_list):
                    resultfile.write("- template: " + ' '.join(tmplt) + '\n')
                    resultfile.write("- expected: " + ' '.join(tgt) + '\n')
                    resultfile.write('- got:      ' + ' '.join(hyp) + '\n\n')
        return eval_bleu, template_bleu

    def _draw_log(epoch, loss_list, test_bleu, tplt_bleu):
        plt.figure(figsize=(14, 10))
        plt.plot(loss_list, '--', linewidth=1, label='loss trend')
        plt.ylabel('training loss till epoch {}'.format(epoch))
        plt.xlabel('every 50 steps')
        plt.savefig(args.log_dir + '/img/train_loss_curve.png')

        plt.figure(figsize=(14, 10))
        plt.plot(test_bleu, '--', linewidth=1, label='test bleu')
        plt.plot(tplt_bleu, '--', linewidth=1, label='template bleu')
        plt.ylabel('bleu till epoch {}'.format(epoch))
        plt.xlabel('every epoch')
        plt.legend(['test bleu', 'template bleu'], loc='upper left')
        plt.savefig(args.log_dir + '/img/bleu.png')
        plt.close('all')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # eval_saver.restore(sess, args.log_dir + 'my-model-highest_bleu.ckpt')
        lowest_loss, highest_bleu, best_epoch = -1, -1, -1
        loss_list, test_bleu, tplt_bleu = [], [], []
        if args.running_mode == 'train_and_evaluate':
            for epoch in range(args.max_train_epoch):
                losses = _train_epochs(sess, epoch)
                test_score, tplt_score = _test_epoch(sess, epoch)
                loss_list.extend(losses)
                test_bleu.append(test_score)
                tplt_bleu.append(tplt_score)
                _draw_log(epoch, loss_list, test_bleu, tplt_bleu)
                if highest_bleu < 0 or test_score > highest_bleu:
                    print('the %d epoch, highest bleu %f' % (epoch, test_score))
                    eval_saver.save(sess, args.log_dir + 'my-model-highest_bleu.ckpt')
                    highest_bleu = test_score
                sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run(main=_main)