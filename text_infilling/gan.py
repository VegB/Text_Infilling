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
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
import texar as tx
import numpy as np
from texar.data import SpecialTokens
from texar.modules.embedders import position_embedders
from texar.utils.shapes import shape_list

import gan_hyperparams
import bleu_tool


def _main(_):
    hparams = baseline_hyperparams.load_hyperparams()
    train_dataset_hparams, valid_dataset_hparams, test_dataset_hparams, encoder_hparams, \
    decoder_hparams, classifier_hparams, opt_hparams, loss_hparams, d_opt_hparams, args = \
        hparams['train_dataset_hparams'], hparams['eval_dataset_hparams'], \
        hparams['test_dataset_hparams'], hparams['encoder_hparams'], hparams['decoder_hparams'], \
        hparams['classifier_hparams'], hparams['opt_hparams'], \
        hparams['loss_hparams'], hparams['d_opt'], hparams['args']

    # Data
    train_data = tx.data.MonoTextData(train_dataset_hparams)
    valid_data = tx.data.MonoTextData(valid_dataset_hparams)
    test_data = tx.data.MonoTextData(test_dataset_hparams)
    iterator = tx.data.FeedableDataIterator(
        {'train_g': train_data, 'train_d': train_data,
         'val': valid_data, 'test': test_data})

    data_batch = iterator.get_next()
    mask_id = train_data.vocab.token_to_id_map_py['<m>']
    boa_id = train_data.vocab.token_to_id_map_py['<BOA>']
    eoa_id = train_data.vocab.token_to_id_map_py['<EOA>']
    eos_id = train_data.vocab.token_to_id_map_py[SpecialTokens.EOS]
    pad_id = train_data.vocab.token_to_id_map_py['<PAD>']
    template_pack, answer_packs = \
        tx.utils.prepare_template(data_batch, args, mask_id, boa_id, eoa_id, pad_id)

    gamma = tf.placeholder(dtype=tf.float32, shape=[], name='gamma')
    lambda_g = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_g')

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

    start_tokens = tf.ones_like(data_batch['length']) * boa_id
    gumbel_helper = tx.modules.GumbelSoftmaxEmbeddingHelper(
        embedder.embedding, start_tokens, eoa_id, gamma)

    # Creates classifier
    classifier = tx.modules.Conv1DClassifier(hparams=classifier_hparams)
    clas_embedder = tx.modules.WordEmbedder(vocab_size=train_data.vocab.size,
                                            hparams=args.word_embedding_hparams)

    cetp_loss, d_class_loss, g_class_loss = None, None, None
    cur_template_pack = template_pack
    for idx, hole in enumerate(answer_packs):
        template = cur_template_pack['templates']
        template_word_embeds = embedder(template)
        template_length = shape_list(template)[1]
        channels = shape_list(template_word_embeds)[2]
        template_pos_embeds = position_embedder(template_length, channels,
                                                cur_template_pack['segment_ids'],
                                                cur_template_pack['offsets'])
        enc_input_embedded = template_word_embeds + template_pos_embeds

        _, ecdr_states = encoder(
            enc_input_embedded,
            sequence_length=data_batch["length"])

        dcdr_init_states = connector(ecdr_states)

        dec_input = hole['text_ids'][:, :-1]
        dec_input_word_embeds = embedder(dec_input)
        decoder.set_segment_id(1)
        dec_input_embedded = dec_input_word_embeds
        outputs, _, _ = decoder(
            initial_state=dcdr_init_states,
            decoding_strategy="train_greedy",
            inputs=dec_input_embedded,
            sequence_length=hole["lengths"] + 1)
        cur_loss = tx.utils.smoothing_cross_entropy(
            outputs.logits,
            hole['text_ids'][:, 1:],
            train_data.vocab.size,
            loss_hparams['label_confidence'],
        )
        cetp_loss = cur_loss if cetp_loss is None \
            else tf.concat([cetp_loss, cur_loss], -1)

        soft_outputs_, _, soft_length_, = decoder(
            helper=gumbel_helper, initial_state=dcdr_init_states)

        # Classification loss for the classifier
        clas_logits, clas_preds = classifier(
            inputs=clas_embedder(ids=hole['text_ids'][:, 1:]),
            sequence_length=hole["lengths"]+1)
        loss_d_clas = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(tf.ones_like(data_batch['length'])), logits=clas_logits)
        d_class_loss = loss_d_clas if d_class_loss is None \
            else tf.concat([d_class_loss, loss_d_clas], -1)

        # Classification loss for the generator, based on soft samples
        soft_logits, soft_preds = classifier(
            inputs=clas_embedder(soft_ids=soft_outputs_.sample_id),
            sequence_length=soft_length_)
        loss_g_clas = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(tf.zeros_like(data_batch['length'])), logits=soft_logits)
        g_class_loss = loss_g_clas if g_class_loss is None \
            else tf.concat([g_class_loss, loss_g_clas], -1)

        cur_template_pack = tx.utils.update_template_pack(cur_template_pack,
                                                          hole['text_ids'][:, 1:],
                                                          mask_id, eoa_id, pad_id)
    cetp_loss = tf.reduce_mean(cetp_loss)
    d_class_loss = tf.reduce_mean(d_class_loss)
    g_class_loss = tf.reduce_mean(g_class_loss)

    global_step = tf.Variable(0, trainable=False)
    if args.learning_rate_strategy == 'static':
        learning_rate = tf.Variable(1e-3, dtype=tf.float32)
    elif args.learning_rate_strategy == 'dynamic':
        fstep = tf.to_float(global_step)
        learning_rate = opt_hparams['lr_constant'] \
                        * args.hidden_dim ** -0.5 \
                        * tf.minimum(fstep ** -0.5, fstep * opt_hparams['warmup_steps'] ** -1.5)
    else:
        raise ValueError('Unknown learning_rate_strategy: %s, expecting one of '
                         '[\'static\', \'dynamic\']' % args.learning_rate_strategy)

    g_loss = cetp_loss + lambda_g * g_class_loss
    g_vars = tx.utils.collect_trainable_variables(
        [embedder, encoder, connector, decoder])
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=opt_hparams['Adam_beta1'],
        beta2=opt_hparams['Adam_beta2'],
        epsilon=opt_hparams['Adam_epsilon'],
    )
    train_op = optimizer.minimize(g_loss, global_step, var_list=g_vars)

    d_loss = d_class_loss
    d_vars = tx.utils.collect_trainable_variables([clas_embedder, classifier])
    train_op_d = tx.core.get_train_op(d_loss, d_vars, hparams=d_opt_hparams)

    # Inference
    predictions = []
    cur_test_pack = template_pack
    for idx, hole in enumerate(answer_packs):
        template = cur_test_pack['templates']
        template_word_embeds = embedder(template)
        template_length = shape_list(template)[1]
        channels = shape_list(template_word_embeds)[2]
        template_pos_embeds = position_embedder(template_length, channels,
                                                cur_test_pack['segment_ids'],
                                                cur_test_pack['offsets'])
        enc_input_embedded = template_word_embeds + template_pos_embeds

        _, ecdr_states = encoder(
            enc_input_embedded,
            sequence_length=data_batch["length"])

        dcdr_init_states = connector(ecdr_states)

        decoder.set_segment_id(1)
        outputs_infer, _, _ = decoder(
            decoding_strategy="infer_positional",
            start_tokens=start_tokens,
            end_token=eoa_id,
            embedding=embedder,
            initial_state=dcdr_init_states)
        predictions.append(outputs_infer.sample_id)
        cur_test_pack = tx.utils.update_template_pack(cur_test_pack,
                                                      outputs_infer.sample_id,
                                                      mask_id, eoa_id, pad_id)

    eval_saver = tf.train.Saver(max_to_keep=5)

    def _train_epochs(session, cur_epoch, gamma_, lambda_g_):
        loss_lists, ppl_lists = [], []
        while True:
            try:
                fetches_d = {
                    'train_op_d': train_op_d,
                    'd_loss': d_loss
                }
                feed_d = {
                    iterator.handle: iterator.get_handle(sess, 'train_d'),
                    gamma: gamma_,
                    lambda_g: lambda_g_,
                    tx.context.global_mode(): tf.estimator.ModeKeys.TRAIN
                }
                rtns_d = session.run(fetches_d, feed_dict=feed_d)
                d_loss_ = rtns_d['d_loss']
                fetches_g = {
                    'template': template_pack,
                    'holes': answer_packs,
                    'train_op': train_op,
                    'step': global_step,
                    'lr': learning_rate,
                    'loss': cetp_loss,
                    'g_loss': g_loss
                }
                feed_g = {
                    iterator.handle: iterator.get_handle(sess, 'train_g'),
                    gamma: gamma_,
                    lambda_g: lambda_g_,
                    tx.context.global_mode(): tf.estimator.ModeKeys.TRAIN
                }
                rtns = session.run(fetches_g, feed_dict=feed_g)
                step, template_, holes_, cetp_loss_, g_loss_ = \
                    rtns['step'], rtns['template'], rtns['holes'], rtns['loss'], rtns['g_loss']
                ppl = np.exp(cetp_loss_)
                if step % 200 == 1:
                    rst = 'step:%s source:%s g_loss:%f d_loss:%f ppl:%f lr:%f' % \
                          (step, template_['text_ids'].shape, g_loss_, d_loss_, ppl, rtns['lr'])
                    print(rst)
                loss_lists.append(g_loss_)
                ppl_lists.append(ppl)
            except tf.errors.OutOfRangeError:
                break
        return loss_lists[::50], ppl_lists[::50]

    def _test_epoch(cur_sess, cur_epoch, gamma_, lambda_g_, mode='test'):
        def _id2word_map(id_arrays):
            return [' '.join([train_data.vocab._id_to_token_map_py[i]
                              for i in sent]) for sent in id_arrays]

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
                feed = {
                    iterator.handle: iterator.get_handle(sess, mode),
                    gamma: gamma_,
                    lambda_g: lambda_g_,
                    tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
                }
                rtns = cur_sess.run(fetches, feed_dict=feed)
                real_templates_, templates_, targets_, predictions_ = \
                    rtns['template']['templates'], rtns['template']['text_ids'], \
                    rtns['data_batch']['text_ids'], rtns['predictions']
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
        print('epoch:{} {}_bleu:{} template_bleu:{} {}_loss:{} {}_ppl:{} '.
              format(cur_epoch, mode, eval_bleu, template_bleu, mode, avg_loss, mode, avg_ppl))
        os.remove(outputs_tmp_filename)
        os.remove(template_tmp_filename)
        os.remove(refer_tmp_filename)
        if args.save_eval_output:
            result_filename = \
                args.log_dir + 'epoch{}.beam{}.{}.results.bleu{:.3f}' \
                    .format(cur_epoch, args.beam_width, mode, eval_bleu)
            with codecs.open(result_filename, 'w+', 'utf-8') as resultfile:
                for tmplt, tgt, hyp in zip(templates_list, targets_list, hypothesis_list):
                    resultfile.write("- template: " + ' '.join(tmplt) + '\n')
                    resultfile.write("- expected: " + ' '.join(tgt) + '\n')
                    resultfile.write('- got:      ' + ' '.join(hyp) + '\n\n')
        return {
            'eval': eval_bleu,
            'template': template_bleu
        }, avg_ppl

    def _draw_train_loss(epoch, loss_list, mode):
        plt.figure(figsize=(14, 10))
        plt.plot(loss_list, '--', linewidth=1, label='loss trend')
        plt.ylabel('%s till epoch %s' % (mode, epoch))
        plt.xlabel('every 50 steps, present_rate=%f' % args.present_rate)
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

    config_ = tf.ConfigProto(allow_soft_placement=True)
    config_.gpu_options.allow_growth = True

    with tf.Session(config=config_) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        iterator.initialize_dataset(sess)

        loss_list, ppl_list, test_ppl_list = [], [], []
        test_bleu, tplt_bleu, train_bleu, train_tplt_bleu = [], [], [], []
        gamma_, lambda_g_ = 1., 0.
        if args.running_mode == 'train_and_evaluate':
            for epoch in range(70, args.max_train_epoch):
                # Anneals the gumbel-softmax temperature
                if epoch > args.pretrain_epoch:
                    gamma_ = max(0.001, gamma_ * args.gamma_decay)
                    lambda_g_ = args.lambda_g

                # bleu on test set and train set
                if epoch % args.bleu_interval == 0 or epoch == args.max_train_epoch - 1:
                    iterator.restart_dataset(sess, 'test')
                    bleu_scores, test_ppl = _test_epoch(sess, epoch, gamma_, lambda_g_)
                    test_bleu.append(bleu_scores['eval'])
                    tplt_bleu.append(bleu_scores['template'])
                    test_ppl_list.append(test_ppl)
                    _draw_train_loss(epoch, test_ppl_list, mode='test_perplexity')

                    iterator.restart_dataset(sess, 'train_g')
                    train_bleu_scores, _ = _test_epoch(sess, epoch, gamma_, lambda_g_, mode='train_g')
                    train_bleu.append(train_bleu_scores['eval'])
                    train_tplt_bleu.append(train_bleu_scores['template'])
                    _draw_bleu(epoch, test_bleu, tplt_bleu, train_bleu, train_tplt_bleu)
                    eval_saver.save(sess, args.log_dir + 'my-model-latest.ckpt')

                # train
                iterator.restart_dataset(sess, ['train_g', 'train_d'])
                losses, ppls = _train_epochs(sess, epoch, gamma_, lambda_g_)
                loss_list.extend(losses)
                ppl_list.extend(ppls)
                _draw_train_loss(epoch, loss_list, mode='train_loss')
                _draw_train_loss(epoch, ppl_list, mode='perplexity')
                sys.stdout.flush()

                if epoch == args.pretrain_epoch:
                    eval_saver.save(sess, args.log_dir + 'pretrained-model.ckpt')


if __name__ == '__main__':
    tf.app.run(main=_main)
