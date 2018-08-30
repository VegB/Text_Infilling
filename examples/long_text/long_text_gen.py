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
    decoder_hparams, opt_hparams, opt_vars, loss_hparams, args = \
        hparams['train_dataset_hparams'], hparams['eval_dataset_hparams'], \
        hparams['test_dataset_hparams'], hparams['decoder_hparams'], \
        hparams['opt_hparams'], hparams['opt_vars'], \
        hparams['loss_hparams'], hparams['args']

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
    train_present_rate = args.present_rate
    test_sets = []
    for rate in args.test_present_rates:
        args.present_rate = rate
        tplt_pack, ans_packs = \
            tx.utils.prepare_template(data_batch, args, mask_id, boa_id, eoa_id, pad_id)
        test_sets.append({
            'test_present_rate': tf.Variable(rate, dtype=tf.float32, trainable=False),
            'template_pack': tplt_pack,
            'answer_packs': ans_packs,
            'predictions': []
        })
    args.present_rate = train_present_rate

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
    learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=opt_hparams['Adam_beta1'],
                                       beta2=opt_hparams['Adam_beta2'],
                                       epsilon=opt_hparams['Adam_epsilon'])
    train_op = optimizer.minimize(cetp_loss, global_step)

    offsets = tx.utils.generate_prediction_offsets(data_batch['text_ids'],
                                                   args.max_decode_len + 1)
    for it, test_pack in enumerate(test_sets):
        for idx, _ in enumerate(test_pack['answer_packs']):
            segment_ids = \
                tx.utils.generate_prediction_segment_ids(data_batch['text_ids'],
                                                     idx * 2 + 1,  # segment id starting from 1
                                                     args.max_decode_len + 1)
            preds = decoder.dynamic_decode(
                template_input_pack=test_pack['template_pack'],
                encoder_decoder_attention_bias=None,
                segment_ids=segment_ids,
                offsets=offsets,
                bos_id=boa_id,
                eos_id=eoa_id)
            test_sets[it]['predictions'].append(preds['sampled_ids'][:, 0])

    def _train_epochs(session, cur_epoch, mode='train'):
        iterator.switch_to_train_data(session)
        loss_lists = []
        cnt = 0
        while True:
            try:
                fetches = {
                    'template': template_pack,
                    'holes': answer_packs,
                    'step': global_step,
                    'loss': cetp_loss
                }
                if mode is 'train':
                    fetches['train_op'] = train_op
                feed = {
                    learning_rate: opt_vars['learning_rate'],
                    tx.context.global_mode(): tf.estimator.ModeKeys.TRAIN if mode is 'train'
                                                else tf.estimator.ModeKeys.EVAL
                }
                rtns = session.run(fetches, feed_dict=feed)
                step, template_, holes_, loss = rtns['step'], \
                    rtns['template'], rtns['holes'], rtns['loss']
                if step % 200 == 1 and mode is 'train':
                    rst = 'step:%s source:%s loss:%s lr:%f' % \
                          (step, template_['text_ids'].shape, loss, opt_vars['learning_rate'])
                    print(rst)
                loss_lists.append(loss)
                cnt += 1
                if mode is not 'train' and cnt >= 50:
                    break
            except tf.errors.OutOfRangeError:
                avg_loss = np.average(loss_list)
                if avg_loss < opt_vars['best_train_loss']:
                    opt_vars['best_train_loss'] = avg_loss
                    opt_vars['epochs_not_improved'] = 0
                else:
                    opt_vars['steps_not_improved'] += 1
                if opt_vars['steps_not_improved'] >= 1:
                    opt_vars['learning_rate'] *= opt_vars['lr_decay_rate']
                break
        return loss_lists

    def _test_epoch(cur_sess, cur_epoch, mode='test'):
        def _id2word_map(id_arrays):
            return [' '.join([train_data.vocab._id_to_token_map_py[i]
                              for i in sent]) for sent in id_arrays]

        test_results = [{'test_present_rate': rate, 'templates_list': [], 'hypothesis_list': []} 
                        for rate in args.test_present_rates]
        targets_list = []
        if mode is 'test':
            iterator.switch_to_test_data(cur_sess)
        elif mode is 'train':
            iterator.switch_to_train_data(cur_sess)
        else:
            iterator.switch_to_val_data(cur_sess)
        cnt = 0
        while True:
            try:
                fetches = {
                    'data_batch': data_batch,
                    'test_sets': test_sets,
                    'step': global_step,
                }
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.EVAL}
                rtns = cur_sess.run(fetches, feed_dict=feed)
                targets_, test_sets_ = rtns['data_batch']['text_ids'], rtns['test_sets']
                targets = _id2word_map(targets_)
                targets_list.extend([target.split('<EOS>')[0].strip().split() for target in targets])

                for it, test_pack in enumerate(test_sets_):
                    templates_list, hypotheses_list = [], []
                    filled_templates = \
                        tx.utils.fill_template(template_pack=test_pack['template_pack'],
                                               predictions=test_pack['predictions'],
                                               eoa_id=eoa_id, pad_id=pad_id, eos_id=eos_id)
                    templates = _id2word_map(test_pack['template_pack']['templates'].tolist())
                    generateds = _id2word_map(filled_templates)
    
                    for template, generated in zip(templates, generateds):
                        template = template.split('<EOS>')[0].strip().split()
                        got = generated.split('<EOS>')[0].strip().split()
                        templates_list.append(template)
                        hypotheses_list.append(got)

                    test_results[it]['templates_list'].extend(templates_list)
                    test_results[it]['hypothesis_list'].extend(hypotheses_list)
                cnt += 1
                if mode is not 'test' and cnt >= 60:
                    break
            except tf.errors.OutOfRangeError:
                break

        bleu_score = [{'test_present_rate': rate} for rate in args.test_present_rates]
        refer_tmp_filename = os.path.join(args.log_dir, 'eval_reference.tmp')
        with codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
            tmpreffile.write('\n'.join([' '.join(tgt) for tgt in targets_list]))
        for it, test_pack in enumerate(test_results):
            outputs_tmp_filename = args.log_dir + 'epoch{}.outputs.tmp'.format(cur_epoch)
            template_tmp_filename = args.log_dir + 'epoch{}.templates.tmp'.format(cur_epoch)
            with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                codecs.open(template_tmp_filename, 'w+', 'utf-8') as tmptpltfile:
                for hyp, tplt in zip(test_pack['hypothesis_list'], test_pack['templates_list']):
                    tmpfile.write(' '.join(hyp) + '\n')
                    tmptpltfile.write(' '.join(tplt) + '\n')
            test_bleu = float(100 * bleu_tool.bleu_wrapper(
                refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
            template_bleu = float(100 * bleu_tool.bleu_wrapper(
                refer_tmp_filename, template_tmp_filename, case_sensitive=True))
            bleu_score[it]['test_bleu'] = test_bleu
            bleu_score[it]['template_bleu'] = template_bleu
            os.remove(outputs_tmp_filename)
            os.remove(template_tmp_filename)
            
            if args.save_eval_output and mode is not 'eval':
                print('epoch:{} test_present_rate:{} {}_bleu:{} template_bleu:{}'
                      .format(cur_epoch, test_pack['test_present_rate'], mode, test_bleu, template_bleu))
                result_filename = \
                    args.log_dir + 'epoch{}.train_present{}.test_present{}.{}.results.bleu{:.3f}'\
                        .format(cur_epoch, args.present_rate, test_pack['test_present_rate'], mode, test_bleu)
                with codecs.open(result_filename, 'w+', 'utf-8') as resultfile:
                    for tmplt, tgt, hyp in zip(test_pack['templates_list'], targets_list, test_pack['hypothesis_list']):
                        resultfile.write("- template: " + ' '.join(tmplt) + '\n')
                        resultfile.write("- expected: " + ' '.join(tgt) + '\n')
                        resultfile.write('- got:      ' + ' '.join(hyp) + '\n\n')
            os.remove(refer_tmp_filename)
        return bleu_score

    def _draw_train_loss(epoch, loss_list):
        plt.figure(figsize=(14, 10))
        plt.plot(loss_list, '--', linewidth=1, label='loss trend')
        plt.ylabel('training loss till epoch {}'.format(epoch))
        plt.xlabel('every 50 steps, present_rate=%f' % args.present_rate)
        plt.savefig(args.log_dir + '/img/train_loss_curve.png')
        plt.close('all')

    def _draw_bleu(epoch, test_bleu, tplt_bleu, train_bleu, train_tplt_bleu):
        plt.figure(figsize=(14, 10))
        legends = []
        for rate in args.test_present_rates:
            plt.plot(test_bleu[rate], '--', linewidth=1, label='test bleu, test pr=%f' % rate)
            plt.plot(tplt_bleu[rate], '--', linewidth=1, label='template bleu, test pr=%f' % rate)
            legends.extend(['test bleu, test pr=%f' % rate, 'template bleu, test pr=%f' % rate])
        plt.ylabel('bleu till epoch {}'.format(epoch))
        plt.xlabel('every epoch, train present rate=%f' % args.present_rate)
        plt.legend(legends, loc='upper left')
        plt.savefig(args.log_dir + '/img/bleu.png')

        plt.figure(figsize=(14, 10))
        legends = []
        for rate in args.test_present_rates:
            plt.plot(train_bleu[rate], '--', linewidth=1, label='train bleu, test pr=%f' % rate)
            plt.plot(train_tplt_bleu[rate], '--', linewidth=1, label='train template bleu, test pr=%f' % rate)
            legends.extend(['train bleu, test pr=%f' % rate, 'train template bleu, test pr=%f' % rate])
        plt.ylabel('bleu till epoch {}'.format(epoch))
        plt.xlabel('every epoch, train present rate=%f' % args.present_rate)
        plt.legend(legends, loc='upper left')
        plt.savefig(args.log_dir + '/img/train_bleu.png')
        plt.close('all')

    eval_saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        """var_list = tf.trainable_variables()
        with open(args.log_dir + 'var.list', 'w+') as outfile:
            for var in var_list:
                outfile.write('var:{} shape:{} dtype:{}\n'.format(\
                    var.name, var.shape, var.dtype))
        total_var_num = 0
        var_list = sess.run(var_list)
        for var in var_list:
            total_var_num += var.size
        print("Total variable number: ", total_var_num)"""

        max_test_bleu = -1
        loss_list, test_bleu, tplt_bleu = [], {rate: [] for rate in args.test_present_rates}, \
                                          {rate: [] for rate in args.test_present_rates}
        train_bleu, train_tplt_bleu = {rate: [] for rate in args.test_present_rates}, \
                                      {rate: [] for rate in args.test_present_rates}
        if args.running_mode == 'train_and_evaluate':
            for epoch in range(args.max_train_epoch):
                # bleu on test set and train set
                if epoch % args.bleu_interval == 0:
                    bleu_scores = _test_epoch(sess, epoch)
                    for scores in bleu_scores:
                        test_bleu[scores['test_present_rate']].append(scores['test_bleu'])
                        tplt_bleu[scores['test_present_rate']].append(scores['template_bleu'])
                        if scores['test_present_rate'] == args.present_rate \
                                and scores['test_bleu'] > max_test_bleu:
                            max_test_bleu = scores['test_bleu']
                            eval_saver.save(sess, args.log_dir + 'my-model-highest_bleu.ckpt')
                    train_bleu_scores = _test_epoch(sess, epoch, mode='train')
                    for scores in train_bleu_scores:
                        train_bleu[scores['test_present_rate']].append(scores['test_bleu'])
                        train_tplt_bleu[scores['test_present_rate']].append(scores['template_bleu'])
                    _draw_bleu(epoch, test_bleu, tplt_bleu, train_bleu, train_tplt_bleu)
                    
                # train
                losses = _train_epochs(sess, epoch)
                loss_list.extend(losses[::50])
                _draw_train_loss(epoch, loss_list)
                sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run(main=_main)
