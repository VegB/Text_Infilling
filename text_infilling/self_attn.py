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

import self_attn_hyperparams
import bleu_tool


def _main(_):
    hparams = self_attn_hyperparams.load_hyperparams()
    train_dataset_hparams, valid_dataset_hparams, test_dataset_hparams, \
    decoder_hparams, opt_hparams, opt_vars, loss_hparams, args = \
        hparams['train_dataset_hparams'], hparams['eval_dataset_hparams'], \
        hparams['test_dataset_hparams'], hparams['decoder_hparams'], \
        hparams['opt_hparams'], hparams['opt_vars'], \
        hparams['loss_hparams'], hparams['args']

    # Data
    train_data = tx.data.MultiAlignedData(train_dataset_hparams)
    valid_data = tx.data.MultiAlignedData(valid_dataset_hparams)
    test_data = tx.data.MultiAlignedData(test_dataset_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=valid_data,
                                             test=test_data)
    data_batch = iterator.get_next()
    vocab = train_data.vocab('source')
    mask_id = vocab.token_to_id_map_py['<m>']
    boa_id = vocab.token_to_id_map_py['<BOA>']
    eoa_id = vocab.token_to_id_map_py['<EOA>']
    eos_id = vocab.token_to_id_map_py['<EOS>']
    pad_id = vocab.token_to_id_map_py['<PAD>']
    template_pack, answer_packs = \
        tx.utils.prepare_template(data_batch, args, mask_id, pad_id)

    # Model architecture
    embedder = tx.modules.WordEmbedder(vocab_size=vocab.size,
                                       hparams=args.word_embedding_hparams)
    decoder = \
        tx.modules.TemplateTransformerDecoder(embedding=embedder._embedding,
                                              hparams=decoder_hparams)

    cetp_loss = None
    cur_template_pack = template_pack
    for hole in answer_packs:
        logits, preds = decoder(decoder_input_pack=hole,
                                template_input_pack=cur_template_pack,
                                encoder_decoder_attention_bias=None,
                                args=args)
        cur_loss = tx.utils.smoothing_cross_entropy(
            logits,
            hole['text_ids'][:, 1:],
            vocab.size,
            loss_hparams['label_confidence'])
        cetp_loss = cur_loss if cetp_loss is None \
            else tf.concat([cetp_loss, cur_loss], -1)
        cur_template_pack = tx.utils.update_template_pack(cur_template_pack,
                                                          hole['text_ids'][:, 1:],
                                                          mask_id, eoa_id, pad_id)
    cetp_loss = tf.reduce_mean(cetp_loss)

    global_step = tf.Variable(0, trainable=False)
    if args.learning_rate_strategy == 'static':
        learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    elif args.learning_rate_strategy == 'dynamic':
        fstep = tf.to_float(global_step)
        learning_rate = opt_hparams['lr_constant'] \
                        * args.hidden_dim ** -0.5 \
                        * tf.minimum(fstep ** -0.5, fstep * opt_hparams['warmup_steps'] ** -1.5)
    else:
        raise ValueError('Unknown learning_rate_strategy: %s, expecting one of '
                         '[\'static\', \'dynamic\']' % args.learning_rate_strategy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=opt_hparams['Adam_beta1'],
                                       beta2=opt_hparams['Adam_beta2'],
                                       epsilon=opt_hparams['Adam_epsilon'])
    train_op = optimizer.minimize(cetp_loss, global_step)

    offsets = tx.utils.generate_prediction_offsets(data_batch['templatebyword_text_ids'],
                                                   args.max_decode_len + 1)
    predictions = []
    cur_test_pack = template_pack
    for idx, hole in enumerate(answer_packs):
        segment_ids = \
            tx.utils.generate_prediction_segment_ids(data_batch['templatebyword_text_ids'],
                                                     1,  # segment_id will always be 1
                                                     args.max_decode_len + 1)
        preds = decoder.dynamic_decode(
            template_input_pack=cur_test_pack,
            encoder_decoder_attention_bias=None,
            segment_ids=segment_ids,
            offsets=offsets,
            bos_id=boa_id,
            eos_id=eoa_id)
        predictions.append(preds['sampled_ids'][:, 0])
        cur_test_pack = tx.utils.update_template_pack(cur_test_pack,
                                                      preds['sampled_ids'][:, 0],
                                                      mask_id, eoa_id, pad_id)

    def _train_epochs(session, cur_epoch, mode='train'):
        iterator.switch_to_train_data(session)
        loss_lists, ppl_lists = [], []
        cnt = 0
        while True:
            try:
                fetches = {
                    'template': template_pack,
                    'holes': answer_packs,
                    'step': global_step,
                    'lr': learning_rate,
                    'loss': cetp_loss
                }
                if mode == 'train':
                    fetches['train_op'] = train_op
                feed = {
                    tx.context.global_mode(): tf.estimator.ModeKeys.TRAIN if mode == 'train'
                    else tf.estimator.ModeKeys.EVAL
                }
                if args.learning_rate_strategy == 'static':
                    feed[learning_rate] = opt_vars['learning_rate']
                rtns = session.run(fetches, feed_dict=feed)
                step, template_, holes_, loss = rtns['step'], \
                                                rtns['template'], rtns['holes'], rtns['loss']
                ppl = np.exp(loss)
                if step % 200 == 1 and mode == 'train':
                    rst = 'step:%s source:%s ppl:%f lr:%f' % \
                          (step, template_['text_ids'].shape, ppl, rtns['lr'])
                    print(rst)
                loss_lists.append(loss)
                ppl_lists.append(ppl)
                cnt += 1
                if mode is not 'train' and cnt >= 50:
                    break
            except tf.errors.OutOfRangeError:
                if args.learning_rate_strategy == 'static':
                    avg_loss = np.average(loss_list)
                    if avg_loss < opt_vars['best_train_loss']:
                        opt_vars['best_train_loss'] = avg_loss
                        opt_vars['epochs_not_improved'] = 0
                    else:
                        opt_vars['epochs_not_improved'] += 1
                    if opt_vars['epochs_not_improved'] >= 8 and opt_vars['decay_time'] <= 3:
                        opt_vars['learning_rate'] *= opt_vars['lr_decay_rate']
                        print("[LR DECAY]: lr decay to %f at epoch %d" %
                              (opt_vars['learning_rate'], cur_epoch))
                        opt_vars['decay_time'] += 1
                break
        return loss_lists, ppl_lists

    def _test_epoch(cur_sess, cur_epoch, mode='test'):
        def _id2word_map(id_arrays):
            return [' '.join([vocab._id_to_token_map_py[i]
                              for i in sent]) for sent in id_arrays]

        if mode == 'test':
            iterator.switch_to_test_data(cur_sess)
        elif mode == 'train':
            iterator.switch_to_train_data(cur_sess)
        else:
            iterator.switch_to_val_data(cur_sess)
        templates_list, targets_list, hypothesis_list = [], [], []
        cnt = 0
        loss_lists, ppl_lists = [], []
        while True:
            try:
                fetches = {
                    'data_batch': data_batch,
                    'predictions': predictions,
                    'template': template_pack,
                    'step': global_step,
                    'loss': cetp_loss
                }
                feed = {tx.context.global_mode(): tf.estimator.ModeKeys.EVAL}
                rtns = cur_sess.run(fetches, feed_dict=feed)
                real_templates_, templates_, targets_, predictions_ = \
                    rtns['template']['templates'], rtns['template']['text_ids'], \
                    rtns['data_batch']['source_text_ids'], rtns['predictions']
                loss = rtns['loss']
                ppl = np.exp(loss)
                loss_lists.append(loss)
                ppl_lists.append(ppl)

                filled_templates = \
                    tx.utils.fill_template(template_pack=rtns['template'],
                                           predictions=rtns['predictions'],
                                           eoa_id=eoa_id, pad_id=pad_id, eos_id=eos_id)

                templates, targets, generateds = _id2word_map(real_templates_.tolist()), \
                                                 _id2word_map(targets_), \
                                                 _id2word_map(filled_templates)

                for template, target, generated in zip(templates, targets, generateds):
                    template = template.split('<EOS>')[0].split('<PAD>')[0].strip().split()
                    target = target.split('<EOS>')[0].split('<PAD>')[0].strip().split()
                    got = generated.split('<EOS>')[0].split('<PAD>')[0].strip().split()
                    templates_list.append(template)
                    targets_list.append(target)
                    hypothesis_list.append(got)

                cnt += 1
                if mode is not 'test' and cnt >= 60:
                    break
            except tf.errors.OutOfRangeError:
                break

        avg_loss, avg_ppl = np.mean(loss_lists), np.mean(ppl_lists)
        outputs_tmp_filename = args.log_dir + 'epoch{}.beam{}.outputs.tmp'. \
            format(cur_epoch, args.beam_width)
        template_tmp_filename = args.log_dir + 'epoch{}.beam{}.templates.tmp'. \
            format(cur_epoch, args.beam_width)
        refer_tmp_filename = os.path.join(args.log_dir, 'eval_reference.tmp')
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
            codecs.open(template_tmp_filename, 'w+', 'utf-8') as tmptpltfile, \
            codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
            for hyp, tplt, tgt in zip(hypothesis_list, templates_list, targets_list):
                tmpfile.write(' '.join(hyp) + '\n')
                tmptpltfile.write(' '.join(tplt) + '\n')
                tmpreffile.write(' '.join(tgt) + '\n')
        eval_bleu = float(100 * bleu_tool.bleu_wrapper(
            refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
        template_bleu = float(100 * bleu_tool.bleu_wrapper(
            refer_tmp_filename, template_tmp_filename, case_sensitive=True))
        print('epoch:{} {}_bleu:{} template_bleu:{} {}_loss:{} {}_ppl:{}'.
              format(cur_epoch, mode, eval_bleu, template_bleu, mode, avg_loss, mode, avg_ppl))
        os.remove(outputs_tmp_filename)
        os.remove(template_tmp_filename)
        os.remove(refer_tmp_filename)
        if args.save_eval_output:
            result_filename = \
                args.log_dir + 'epoch{}.beam{}.{}.results.bleu{:.3f}'\
                    .format(cur_epoch, args.beam_width, mode, eval_bleu)
            with codecs.open(result_filename, 'w+', 'utf-8') as resultfile:
                for tmplt, tgt, hyp in zip(templates_list, targets_list, hypothesis_list):
                    resultfile.write("- template: " + ' '.join(tmplt) + '\n')
                    resultfile.write("- expected: " + ' '.join(tgt) + '\n')
                    resultfile.write('- got:      ' + ' '.join(hyp) + '\n\n')
        return {
            'eval': eval_bleu,
            'template': template_bleu
        }

    def _draw_train_loss(epoch, loss_list, mode):
        plt.figure(figsize=(14, 10))
        plt.plot(loss_list, '--', linewidth=1, label='loss trend')
        plt.ylabel('%s till epoch %s' % (mode, epoch))
        plt.xlabel('every 50 steps')
        plt.savefig(args.log_dir + '/img/%s_curve.png' % mode)
        plt.close('all')

    def _draw_bleu(epoch, test_bleu, tplt_bleu, train_bleu, train_tplt_bleu):
        plt.figure(figsize=(14, 10))
        legends = []
        plt.plot(test_bleu, '--', linewidth=1, label='test bleu')
        plt.plot(tplt_bleu, '--', linewidth=1, label='template bleu')
        legends.extend(['test bleu', 'template bleu'])
        plt.ylabel('bleu till epoch {}'.format(epoch))
        plt.xlabel('every epoch')
        plt.legend(legends, loc='upper left')
        plt.savefig(args.log_dir + '/img/bleu.png')

        plt.figure(figsize=(14, 10))
        legends = []
        plt.plot(train_bleu, '--', linewidth=1, label='train bleu')
        plt.plot(train_tplt_bleu, '--', linewidth=1, label='train template bleu')
        legends.extend(['train bleu', 'train template bleu'])
        plt.ylabel('bleu till epoch {}'.format(epoch))
        plt.xlabel('every epoch')
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

        loss_list, ppl_list, test_ppl_list = [], [], []
        test_bleu, tplt_bleu, train_bleu, train_tplt_bleu = [], [], [], []
        if args.running_mode == 'train_and_evaluate':
            for epoch in range(args.max_train_epoch):
                # bleu on test set and train set
                if epoch % args.bleu_interval == 0 or epoch == args.max_train_epoch - 1:
                    bleu_scores = _test_epoch(sess, epoch)
                    test_bleu.append(bleu_scores['eval'])
                    tplt_bleu.append(bleu_scores['template'])
                    """train_bleu_scores = _test_epoch(sess, epoch, mode='train')
                    train_bleu.append(train_bleu_scores['eval'])
                    train_tplt_bleu.append(train_bleu_scores['template'])"""
                    _draw_bleu(epoch, test_bleu, tplt_bleu, train_bleu, train_tplt_bleu)
                    eval_saver.save(sess, args.log_dir + 'my-model-latest.ckpt')

                # train
                losses, ppls = _train_epochs(sess, epoch)
                loss_list.extend(losses)
                ppl_list.extend(ppls)
                _draw_train_loss(epoch, loss_list, mode='train_loss')
                _draw_train_loss(epoch, ppl_list, mode='perplexity')
                _draw_train_loss(epoch, test_ppl_list, mode='test_perplexity')
                sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run(main=_main)

