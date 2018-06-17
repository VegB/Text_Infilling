from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR
import sys
import time
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

flags = tf.flags

flags.DEFINE_string("data_path", "./", "Directory containing PTB. If not exists, "
                    "the directory will be created, and the data will be downloaded.")
flags.DEFINE_string("config", "config", "The config to use.")
FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)
log = open(config.log_file, 'w')

def prepare_data(train_path):
    """Download the PTB or Yahoo dataset
    """
    ptb_url = 'https://jxhe.github.io/download/ptb_data.tgz'

    data_path = FLAGS.data_path

    if not tf.gfile.Exists(train_path):
        url = ptb_url
        tx.data.maybe_download(url, data_path, extract=True)
        os.remove('ptb_data.tgz')

        data_path = os.path.join(data_path, '%s_data' % FLAGS.dataset)

        train_path = os.path.join(data_path, "%s.train.txt" % FLAGS.dataset)
        valid_path = os.path.join(data_path, "%s.valid.txt" % FLAGS.dataset)
        test_path = os.path.join(data_path, "%s.test.txt" % FLAGS.dataset)
        vocab_path = os.path.join(data_path, "vocab.txt")

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

    batch_size = tf.shape(data_batch["text_ids"])[0]
    num_steps = tf.shape(data_batch["text_ids"])[1]
    vocab_size = train_data.vocab.size
    opt_vars = {
        'learning_rate': config.lr_hparams['init_lr'],
        'update_learning_rate': config.lr_hparams['update_lr'],
        'best_valid_ppl': 1e100,
        'steps_not_improved': 0
    }

    # Model architecture
    g_embedder = tx.modules.WordEmbedder(vocab_size=vocab_size, hparams=config.emb_hparams)
    input_embed = g_embedder(data_batch["text_ids"][:, :-1])

    if config.enc_keep_prob_in < 1:
        input_embed = tf.nn.dropout(
            input_embed, tx.utils.switch_dropout(config.enc_keep_prob_in))

    decoder = tx.modules.BasicRNNDecoder(vocab_size=vocab_size,
                                         hparams={"rnn_cell": config.dec_cell_hparams})
    initial_state = decoder.zero_state(batch_size=batch_size, dtype=tf.float32)

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

    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0., beta2=0.999, epsilon=1e-9)
    gen_train_op = optimizer.minimize(mle_loss, global_step=global_step)

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
    d_embedder = tx.modules.WordEmbedder(vocab_size=vocab_size, hparams=config.emb_hparams)

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
                                        increment_global_step=False, hparams=config.d_opt_hparams)

    # ------------Adeversarial---------------
    infer_logits = \
        tf.clip_by_value(tf.nn.softmax(infer_logits) * tf.one_hot(infer_sample_ids, vocab_size), 1e-20, 1)

    expected_reward = tf.Variable(tf.zeros((config.max_num_steps,)))  # (num_step,), exp_reward at each step
    reward = tf.squeeze(f_logits) - expected_reward[:tf.shape(f_logits)[1]]
    mean_reward = tf.reduce_mean(reward)
    exp_reward_loss = tf.reduce_mean(tf.abs(reward))
    exp_reward_loss.set_shape(())
    exp_op = tx.core.get_train_op(exp_reward_loss, global_step=global_step,
                                  increment_global_step=False, hparams=config.update_opt_hparams)

    reward = tx.losses.discount_reward(reward, sequence_length=tf.squeeze(sequence_length), tensor_rank=2)
    update_loss = -tf.reduce_mean(tf.log(infer_logits) * tf.expand_dims(reward, -1))
    update_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                              beta1=0., beta2=0.999, epsilon=1e-9)
    gen_op = update_optimizer.minimize(update_loss, global_step=global_step)
    update_op = tf.group(gen_op, exp_op)

    def _g_run_epoch(sess, mode_string):
        if mode_string in ['train', 'update']:
            iterator.switch_to_train_data(sess)
        elif mode_string == 'valid':
            iterator.switch_to_val_data(sess)
        elif mode_string == 'test':
            iterator.switch_to_test_data(sess)
        else:
            sys.exit("INVALID MODE %s, expecting one of "
                     "['train', 'valid', 'test', 'update']" % mode_string)

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
        else:
            fetches = {
                "mle_loss": mle_loss,
                'step': global_step,
                "num_steps": num_steps
            }
            if mode_string == 'train':
                fetches["train_op"] = gen_train_op

        loss, iters = 0., 0

        while True:
            try:
                feed_dict = {
                    learning_rate: opt_vars['learning_rate'],
                    tx.global_mode(): tf.estimator.ModeKeys.TRAIN if mode_string in ['train', 'update']
                    else tf.estimator.ModeKeys.EVAL
                }

                rtns = sess.run(fetches, feed_dict)
                iters += rtns["num_steps"]
                ppl = np.exp(loss / iters)

                if mode_string != 'update':
                    loss += rtns["mle_loss"]

                if mode_string in ['train', 'update'] and rtns['step'] % 100 == 0:
                    valid_ppl = _g_run_epoch(sess, 'valid')
                    test_ppl = _g_run_epoch(sess, 'test')
                    if mode_string == 'train':
                        rst = "step: %d, v_ppl: %.6f, tst_ppl: %.6f, tr_ppl: %.6f, lr: %.7f" % \
                              (rtns['step'], valid_ppl, test_ppl, ppl, opt_vars['learning_rate'])
                    else:
                        rst = "step: %d, v_ppl: %.6f, tst_ppl: %.6f, mean_rwd: %.6f, exp_rwd_loss:" \
                              " %.6f, update_loss: %.6f" % (rtns['step'], valid_ppl, test_ppl,
                                                            rtns['mean_rwd'], rtns['exp_rwd_loss'], rtns['update_loss'])
                    log.write(rst + '\n')
                    print(rst)

                    if valid_ppl < opt_vars['best_valid_ppl']:
                        opt_vars['best_valid_ppl'] = valid_ppl
                        opt_vars['steps_not_improved'] = 0
                    else:
                        opt_vars['steps_not_improved'] += 1

                    if opt_vars['steps_not_improved'] >= 30:
                        opt_vars['steps_not_improved'] = 0
                        opt_vars['learning_rate'] *= config.lr_hparams['lr_decay']

            except tf.errors.OutOfRangeError:
                ppl = np.exp(loss / iters)
                break

        return ppl

    def _d_run_epoch(sess):
        fetches = {
            "mle_loss": dis_loss,
            "r_loss": r_loss,
            "f_loss": f_loss,
            "train_op": dis_train_op
        }
        step = 0
        while True:
            try:
                feed_dict = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
                rtns = sess.run(fetches, feed_dict)
                if step % 200 == 0:
                    print("{0:3d}: dis_total_loss: {1:6f}, r_loss: {2:6f}, f_loss: {3:6f}"
                          .format(step, rtns['mle_loss'], rtns['r_loss'], rtns['f_loss']))
                step += 1
            except tf.errors.OutOfRangeError:
                break

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver()
        
        for g_epoch in range(config.generator_pretrain_epoch):
            _g_run_epoch(sess, 'train')
            if (g_epoch + 1) % 20 == 0:
                saver.save(sess, config.ckpt, global_step=g_epoch + 1)
        
        for d_epoch in range(config.discriminator_pretrain_epoch):
            _d_run_epoch(sess)
        saver.save(sess, config.ckpt, global_step=config.generator_pretrain_epoch + 1)

        opt_vars['learning_rate'] = config.lr_hparams['update_init_lr']
        for update_epoch in range(config.adversial_epoch):
            _g_run_epoch(sess, 'update')
            if (update_epoch + 1) % 20 == 0:
                saver.save(sess, config.ckpt,
                           global_step=config.generator_pretrain_epoch + update_epoch + 1)

    log.close()


if __name__ == '__main__':
    tf.app.run(main=_main)
