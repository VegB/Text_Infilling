from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import codecs
import importlib
import numpy as np
import tensorflow as tf
import texar as tx
import bleu_tool
from data_utils import prepare_data

flags = tf.flags
flags.DEFINE_string("dataset", "coco",
                    "perform training on ptb or coco.")
flags.DEFINE_string("data_path", "./",
                    "Directory containing coco. If not exists, "
                    "the directory will be created, and the data "
                    "will be downloaded.")
flags.DEFINE_string("config", "config", "The config to use.")
FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)


def _main(_):
    prepare_data(FLAGS, config,
                 config.train_data_hparams['dataset']['files'])
    log = open(config.log_file, 'w')
    bleu_log = open(config.bleu_file, 'w')

    # Data
    train_data = tx.data.MonoTextData(config.train_data_hparams)
    val_data = tx.data.MonoTextData(config.val_data_hparams)
    test_data = tx.data.MonoTextData(config.test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()

    batch_size = tf.shape(data_batch["text_ids"])[0]
    num_steps = tf.shape(data_batch["text_ids"])[1]
    vocab_size = train_data.vocab.size

    # Model architecture
    g_embedder = tx.modules.WordEmbedder(vocab_size=vocab_size,
                                         hparams=config.emb_hparams)
    input_embed = g_embedder(data_batch["text_ids"][:, :-1])

    if config.enc_keep_prob_in < 1:
        input_embed = tf.nn.dropout(
            input_embed, tx.utils.switch_dropout(config.enc_keep_prob_in))

    decoder = tx.modules.BasicRNNDecoder(
        vocab_size=vocab_size,
        hparams={"rnn_cell": config.dec_cell_hparams,
                 "max_decoding_length_infer": config.max_num_steps + 2})
    initial_state = decoder.zero_state(batch_size=batch_size,
                                       dtype=tf.float32)

    # ------------Pretrain Generator---------------
    outputs, _, _ = decoder(
        initial_state=initial_state,
        decoding_strategy="train_greedy",
        inputs=input_embed,
        sequence_length=data_batch["length"] - 1)

    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch["text_ids"][:, 1:],
        logits=outputs.logits,
        sequence_length=data_batch["length"] - 1)

    global_step = tf.Variable(0, trainable=False)
    gen_train_op = tx.core.get_train_op(mle_loss, global_step=global_step,
                                        increment_global_step=False,
                                        hparams=config.g_opt_hparams)

    # -------------Generator Infer-------------------
    infer_outputs, _, sequence_length = decoder(
        decoding_strategy="infer_sample",
        start_tokens=tf.cast(data_batch["text_ids"][:, 0], dtype=tf.int32),
        end_token=train_data.vocab.eos_token_id,
        embedding=g_embedder,
        initial_state=initial_state,
        max_decoding_length=config.max_num_steps)

    infer_logits = infer_outputs.logits
    infer_sample_ids = infer_outputs.sample_id

    # ------------Pretrain Discriminator---------------
    discriminator = tx.modules.UnidirectionalRNNClassifier(
        hparams={"clas_strategy": "time_wise", "num_classes": 1})
    d_embedder = tx.modules.WordEmbedder(vocab_size=vocab_size,
                                         hparams=config.emb_hparams)

    r_logits, _ = discriminator(d_embedder(data_batch["text_ids"]))
    f_logits, _ = discriminator(d_embedder(infer_sample_ids))

    r_loss = tx.losses.sequence_sigmoid_cross_entropy(
        labels=tf.ones_like(data_batch["text_ids"], dtype=tf.float32),
        logits=tf.squeeze(r_logits),
        sequence_length=data_batch["length"])  # r_preds -> 1.
    f_loss = tx.losses.sequence_sigmoid_cross_entropy(
        labels=tf.zeros_like(infer_sample_ids, dtype=tf.float32),
        logits=tf.squeeze(f_logits),
        sequence_length=sequence_length)  # infer_logits -> 0.
    dis_loss = r_loss + f_loss
    dis_loss.set_shape(())

    dis_train_op = tx.core.get_train_op(dis_loss, global_step=global_step,
                                        increment_global_step=False,
                                        hparams=config.d_opt_hparams)

    # ------------Adeversarial---------------
    infer_logits = \
        tf.clip_by_value(tf.nn.softmax(infer_logits) *
                         tf.one_hot(infer_sample_ids, vocab_size), 1e-20, 1)

    expected_reward = tf.Variable(tf.zeros((config.max_num_steps,)))
    reward = tf.squeeze(f_logits) - expected_reward[:tf.shape(f_logits)[1]]
    mean_reward = tf.reduce_mean(reward)
    exp_reward_loss = tf.reduce_mean(tf.abs(reward))
    exp_reward_loss.set_shape(())
    exp_op = tx.core.get_train_op(exp_reward_loss, global_step=global_step,
                                  increment_global_step=False,
                                  hparams=config.update_opt_hparams)
    reward = \
        tx.losses.discount_reward(reward,
                                  sequence_length=tf.squeeze(sequence_length),
                                  tensor_rank=2)
    update_loss = \
        -tf.reduce_mean(tf.log(infer_logits) * tf.expand_dims(reward, -1))
    update_loss.set_shape(())
    gen_op = tx.core.get_train_op(update_loss, global_step=global_step,
                                  increment_global_step=False,
                                  hparams=config.update_opt_hparams)
    update_op = tf.group(gen_op, exp_op)

    def _g_train_epoch(sess, mode_string):
        iterator.switch_to_train_data(sess)
        step = 0
        while True:
            try:
                if mode_string == 'update':
                    fetches = {
                        "mean_rwd": mean_reward,
                        "exp_rwd_loss": exp_reward_loss,
                        "update_loss": update_loss,
                        "train_op": update_op,
                        "exp_rwd": expected_reward,
                        'step': global_step,
                        "num_steps": num_steps
                    }
                else:  # 'train'
                    fetches = {
                        "mle_loss": mle_loss,
                        'step': global_step,
                        "num_steps": num_steps,
                        'train_op': gen_train_op
                    }
                rtns = sess.run(fetches)
                step += 1
                if step % 100 == 1:
                    if mode_string == 'train':
                        ppl = np.exp(rtns['mle_loss'] / rtns["num_steps"])
                        rst = "step: %d, tr_ppl: %.6f" % (step, ppl)
                    else:
                        rst = "step: %d, mean_rwd: %.6f, exp_rwd_loss:%.6f, " \
                              "update_loss: %.6f" % \
                              (step, rtns['mean_rwd'],
                               rtns['exp_rwd_loss'],
                               rtns['update_loss'])
                    log.write(rst + '\n')
                    log.flush()
                    print(rst)
            except tf.errors.OutOfRangeError:
                break
        return

    def _g_test_epoch(sess, mode_string, epoch):
        def _id2word_map(id_arrays):
            return [' '.join([train_data.vocab._id_to_token_map_py[i]
                              for i in sent]) for sent in id_arrays]

        if mode_string == 'valid':
            iterator.switch_to_val_data(sess)
        elif mode_string == 'test':
            iterator.switch_to_test_data(sess)

        inference_list = []
        while True:
            try:
                fetches = {'infer_sample_id': infer_sample_ids}
                feed_dict = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}
                rtns = sess.run(fetches, feed_dict)
                inferences = _id2word_map(rtns['infer_sample_id'].tolist())
                inference_list.extend([inf.split('<EOS>')[0].strip().split()
                                       for inf in inferences])
            except tf.errors.OutOfRangeError:
                break

        outputs_filename = config.log_dir + 'epoch%d.txt' % epoch
        with codecs.open(outputs_filename, 'w+', 'utf-8') as fout:
            for inf in inference_list:
                fout.write(' '.join(inf) + '\n')
        bleu1, bleu2, bleu3, bleu4 = bleu_tool.calculate_bleu(
            candidate_path=config.test_data_hparams['dataset']['files'],
            reference_path=outputs_filename)
        buf = "epoch %d BLEU1~4:\n%f\n%f\n%f\n%f\n\n" % \
              (epoch, bleu1, bleu2, bleu3, bleu4)
        print(buf)
        bleu_log.write(buf + '\n')
        bleu_log.flush()
        return

    def _d_run_epoch(sess):
        iterator.switch_to_train_data(sess)
        step = 0
        while True:
            try:
                fetches = {
                    "mle_loss": dis_loss,
                    "r_loss": r_loss,
                    "f_loss": f_loss,
                    "train_op": dis_train_op
                }
                rtns = sess.run(fetches)
                if step % 50 == 0:
                    print("{0:3d}: dis_total_loss: {1:6f}, "
                          "r_loss: {2:6f}, f_loss: {3:6f}"
                          .format(step, rtns['mle_loss'],
                                  rtns['r_loss'], rtns['f_loss']))
                step += 1
            except tf.errors.OutOfRangeError:
                break

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        
        for g_epoch in range(config.generator_pretrain_epoch):
            _g_train_epoch(sess, 'train')
            if g_epoch % 10 == 0:
                _g_test_epoch(sess, 'test', g_epoch)

        for d_epoch in range(config.discriminator_pretrain_epoch):
            _d_run_epoch(sess)

        for update_epoch in range(config.adversial_epoch):
            _g_train_epoch(sess, 'update')
            if update_epoch % 10 == 0:
                _g_test_epoch(sess, 'test', update_epoch)
    log.close()
    bleu_log.close()


if __name__ == '__main__':
    tf.app.run(main=_main)
