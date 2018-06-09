import tensorflow as tf
import numpy as np
import importlib
import sys

import texar as tx

from embedding_tied_language_model import EmbeddingTiedLanguageModel
from dataloader import DataLoader
from utils import print_and_write_to_file


def g_run_epoch(sess, mode_string):
    if mode_string == 'train':
        dataloader = train_dataloader
    elif mode_string == 'valid':
        dataloader = valid_dataloader
    elif mode_string == 'test':
        dataloader = test_dataloader
    else:
        sys.exit("INVALID MODE %s, expecting one of ['train', 'valid', 'test']" % mode_string)

    loss = 0.
    iters = 0
    state = sess.run(initial_state, feed_dict={inputs: np.ones((batch_size, num_steps))})

    fetches = {
        "mle_loss": mle_loss,
        'initial_state': initial_state,
        "final_state": final_state,
        'global_step': global_step
    }
    if mode_string == 'train':
        fetches["train_op"] = g_train_op

    for step, (x, y) in enumerate(dataloader.iter()):
        feed_dict = {
            inputs: x, targets: y,
            learning_rate: opt_vars['learning_rate'],
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN if mode_string == 'train'
            else tf.estimator.ModeKeys.EVAL
        }
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        rets = sess.run(fetches, feed_dict)
        loss += rets["mle_loss"]
        state = rets["final_state"]
        iters += num_steps
        ppl = np.exp(loss / iters)

        if mode_string == 'train' and rets['global_step'] % 100 == 0:
            valid_ppl = g_run_epoch(sess, 'valid')
            test_ppl = g_run_epoch(sess, 'test')
            rst = "step: %d, tr_ppl: %.6f, v_ppl: %.6f, tst_ppl: %.6f, lr: %.7f\n" % \
                  (rets['global_step'], ppl, valid_ppl, test_ppl, opt_vars['learning_rate'])
            print_and_write_to_file(rst, eval_log)

            if valid_ppl < opt_vars['best_valid_ppl']:
                opt_vars['best_valid_ppl'] = valid_ppl
                opt_vars['steps_not_improved'] = 0
            else:
                opt_vars['steps_not_improved'] += 1

            if opt_vars['steps_not_improved'] >= 30:
                opt_vars['steps_not_improved'] = 0
                opt_vars['learning_rate'] *= config.lr_decay

    ppl = np.exp(loss / iters)
    return ppl


if __name__ == "__main__":
    config_path = "config"
    config = importlib.import_module(config_path)
    eval_log = open(config.log_hparams['eval_log_file'], "w")

    word2id = tx.data.make_vocab(config.data_hparams["train"], newline_token="<EOS>", return_type="dict")
    vocab_size = len(word2id)
    train_dataloader = DataLoader(config, config.data_hparams["train"], word2id)
    valid_dataloader = DataLoader(config, config.data_hparams["valid"], word2id)
    test_dataloader = DataLoader(config, config.data_hparams["test"], word2id)

    generator = EmbeddingTiedLanguageModel(vocab_size=vocab_size)
    discriminator = tx.modules.UnidirectionalRNNClassifier(hparams={"clas_strategy": "time_wise", "num_classes": 1})

    # ------------Pretrain Generator---------------
    batch_size = config.training_hparams['batch_size']
    num_steps = config.training_hparams['num_steps']

    opt_vars = config.opt_vars

    inputs = tf.placeholder(tf.int32, [None, num_steps])
    targets = tf.placeholder(tf.int32, [None, num_steps])

    initial_state, gen_logits, final_state = \
        generator(text_ids=inputs, num_steps=num_steps * tf.ones((batch_size,), dtype=tf.int32))

    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=targets,
        logits=gen_logits,
        sequence_length=num_steps * tf.ones((batch_size,)))

    l2_loss = sum([tf.nn.l2_loss(t) for t in tf.trainable_variables()])

    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = \
        tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.,
        beta2=0.999,
        epsilon=1e-9)
    g_train_op = optimizer.minimize(
        mle_loss + config.l2_hparams['l2_decay'] * l2_loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver()

        for pre_epoch in range(1, config.training_hparams['generator_pretrain_epoch'] + 1):
            train_ppl = g_run_epoch(sess, 'train')

