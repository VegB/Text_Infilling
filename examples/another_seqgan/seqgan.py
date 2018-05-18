import tensorflow as tf
import numpy as np
import importlib

import texar as tx

from generator import Generator
from discriminator import Discriminator
from dataloader import GenDataLoader, DisDataLoader
from utils import print_result, store_output, pad_to_length, print_and_write_to_file

config_path = "config"
config = importlib.import_module(config_path)
log = open(config.log_file, "a+")
training_log = open(config.train_log_file, "w")
eval_log = open(config.eval_log_file, "w")


def pretrain_generator(sess, gen_dataloader, valid_dataloader, test_dataloader):
    loss = 0.
    iters = 0
    state = sess.run(initial_state, feed_dict={
        inputs: np.ones((config.batch_size, config.num_steps))})

    fetches = {
        "mle_loss": mle_loss,
        "final_state": final_state,
        'global_step': global_step,
        'train_op': train_op
    }

    for step, (x, y) in enumerate(gen_dataloader.iter()):
        feed_dict = {
            batch_size: config.batch_size,
            inputs: x, targets: y,
            learning_rate: opt_vars['learning_rate'],
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
        }
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        rets = sess.run(fetches, feed_dict)
        loss += rets["mle_loss"]
        state = rets["final_state"]
        iters += config.num_steps

        ppl = np.exp(loss / iters)

        rst = "global step: %d, training ppl: %.6f\n" % (rets['global_step'], ppl)
        print_and_write_to_file(rst, training_log, print_out=False)

        if rets['global_step'] % 100 == 0:
            valid_ppl = calculate_ppl(sess, valid_dataloader)
            test_ppl = calculate_ppl(sess, test_dataloader)
            rst = "global step: %d, learning rate: %.7f, training ppl: %.6f, valid ppl: %.6f, test ppl: %.6f\n" % \
                  (rets['global_step'], opt_vars['learning_rate'], ppl, valid_ppl, test_ppl)
            print_and_write_to_file(rst, eval_log)

            # print_result(rets['sample_id'][:config.print_num], gen_dataloader.id2word, gen_dataloader.max_len)

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


def generate_negative_samples(sess, gen_dataloader, dst_path):
    print("-------------Generate Negative Samples----------------")
    generated_outputs = []
    fetches = {
        'generated_sample_id': generated_sample_id
    }
    for step, (x, y) in enumerate(gen_dataloader.iter()):
        feed_dict = {
            batch_size: config.batch_size,
            inputs: x, targets: y,
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
        }
        rets = sess.run(fetches, feed_dict)
        generated_outputs.extend(rets['generated_sample_id'])

    store_output(output=generated_outputs, id2word=gen_dataloader.id2word,
                 data_path=dst_path, max_len=config.num_steps)
    print_result(generated_outputs[:config.print_num], gen_dataloader.id2word, config.num_steps)


def train_discriminator(sess, discriminator, epoch_num):
    print("-------------Train Discriminator----------------")
    dataloader = DisDataLoader(config, positive_file=config.train_file,
                               negative_file=config.train_file, word2id=word2id)

    for step, (r_ids, g_ids) in enumerate(dataloader.iter()):
        _, loss = sess.run([discriminator.train_op, discriminator.total_loss],
                                 feed_dict={discriminator.real_samples: r_ids,
                                            discriminator.gen_samples: g_ids,
                                            discriminator.global_step: step,
                                            tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        if step % 200 == 0:
            print("%d: dis_total_loss: %.6f" % (step, loss))


def update_generator(sess, gen_dataloader):
    print("-------------Update Generator----------------")
    loss, epoch_update_loss = 0., 0.
    iters = 0
    state = sess.run(initial_state, feed_dict={
        inputs: np.ones((config.batch_size, config.num_steps))})

    fetches = {
        "mle_loss": mle_loss,
        'update_loss': update_loss,
        "final_state": final_state,
        'global_step': global_step,
        'train_op': train_op,
        'update_op': update_op
    }

    for step, (x, y) in enumerate(gen_dataloader.iter()):
        feed_dict = {
            batch_size: config.batch_size,
            inputs: x, targets: y,
            learning_rate: opt_vars['learning_rate'],
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
        }
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        rets = sess.run(fetches, feed_dict)
        loss += rets["mle_loss"]
        epoch_update_loss += rets['update_loss']
        state = rets["final_state"]
        iters += config.num_steps

        ppl = np.exp(loss / iters)

        rst = "global step: %d, training ppl: %.6f, update loss: %.6f\n" % \
              (rets['global_step'], ppl, rets['update_loss'])
        print_and_write_to_file(rst, training_log, print_out=False)

        if rets['global_step'] % 100 == 0:
            valid_ppl = calculate_ppl(sess, valid_dataloader)
            test_ppl = calculate_ppl(sess, test_dataloader)
            rst = "global step: %d, learning rate: %.7f, training ppl: %.6f," \
                  " valid ppl: %.6f, test ppl: %.6f, update_loss: %.6f\n " % \
                  (rets['global_step'], opt_vars['learning_rate'], ppl, valid_ppl, test_ppl, epoch_update_loss/iters)
            print_and_write_to_file(rst, eval_log)

            if valid_ppl < opt_vars['best_valid_ppl']:
                opt_vars['best_valid_ppl'] = valid_ppl
                opt_vars['steps_not_improved'] = 0
            else:
                opt_vars['steps_not_improved'] += 1

            if opt_vars['steps_not_improved'] >= 30:
                opt_vars['steps_not_improved'] = 0
                opt_vars['learning_rate'] *= config.lr_decay
                opt_vars['update_learning_rate'] *= config.lr_decay

    ppl = np.exp(loss / iters)
    return ppl


def calculate_ppl(sess, dataloader):
    loss = 0.
    iters = 0
    state = sess.run(initial_state, feed_dict={
        inputs: np.ones((config.batch_size, config.num_steps))})

    fetches = {
        "mle_loss": mle_loss,
        "final_state": final_state,
        'global_step': global_step,
    }

    for step, (x, y) in enumerate(dataloader.iter()):
        feed_dict = {
            batch_size: config.batch_size,
            inputs: x, targets: y,
            learning_rate: opt_vars['learning_rate'],
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
        }
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        rets = sess.run(fetches, feed_dict)
        loss += rets["mle_loss"]
        state = rets["final_state"]
        iters += config.num_steps
    ppl = np.exp(loss / iters)
    return ppl


if __name__ == "__main__":
    word2id = tx.data.make_vocab(config.train_file, newline_token="<EOS>", return_type="dict")
    vocab_size = len(word2id)
    gen_dataloader = GenDataLoader(config, config.train_file, word2id)
    valid_dataloader = GenDataLoader(config, config.valid_file, word2id)
    test_dataloader = GenDataLoader(config, config.test_file, word2id)

    generator = Generator(vocab_size=vocab_size)
    discriminator = Discriminator(config, vocab_size=vocab_size)
    saver = tf.train.Saver()

    # ------------Pretrain---------------
    batch_size = tf.placeholder(dtype=tf.int32, shape=())
    num_steps = config.num_steps

    opt_vars = {
        'learning_rate': config.init_lr,
        'update_learning_rate': config.update_lr,
        'best_valid_ppl': 1e100,
        'steps_not_improved': 0
    }

    inputs = tf.placeholder(tf.int32, [None, num_steps])
    targets = tf.placeholder(tf.int32, [None, num_steps])

    initial_state, logits, final_state, sample_id = \
        generator(text_ids=inputs, num_steps=config.num_steps * tf.ones((batch_size,), dtype=tf.int32))

    # Losses & train ops
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=targets,
        logits=logits,
        sequence_length=num_steps * tf.ones((batch_size,)))

    l2_loss = sum([tf.nn.l2_loss(t) for t in tf.trainable_variables()])

    # Use global_step to pass epoch, for lr decay
    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = \
        tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.,
        beta2=0.999,
        epsilon=1e-9)
    train_op = optimizer.minimize(
        mle_loss + config.l2_decay * l2_loss, global_step=global_step)

    generated_outputs, _, _ = generator.decoder(
        decoding_strategy="infer_sample",
        start_tokens=inputs[:, 0],
        end_token=word2id["<EOS>"],
        embedding=generator.embedding_matrix,
        initial_state=initial_state,
        max_decoding_length=num_steps)
    generated_logits = generator.output_layer(generated_outputs.logits)
    generated_sample_id = tf.argmax(generated_logits, 2)

    # ----------------Adversarial------------------
    rewards = discriminator(data=sample_id)

    preds = tf.nn.softmax(logits)
    update_loss = -tf.reduce_sum(
        tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(inputs, [-1])), vocab_size,
                       1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(preds[:, :config.num_steps, :], [-1, vocab_size]), 1e-20, 1.0)
            ), 1) * tf.reshape(rewards, [-1])
    )

    update_step = tf.Variable(0, dtype=tf.int32)
    update_optimizer = tf.train.AdamOptimizer(
        learning_rate=opt_vars['update_learning_rate'],
        beta1=0.,
        beta2=0.999,
        epsilon=1e-9)
    update_op = update_optimizer.minimize(update_loss, global_step=global_step)
    '''update_op = tx.core.get_train_op(
        update_loss, global_step=update_step, increment_global_step=False,
        hparams=config.opt)'''
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        for pre_epoch in range(1, config.generator_pretrain_epoch + 1):
            train_ppl = pretrain_generator(sess, gen_dataloader, valid_dataloader, test_dataloader)
            if pre_epoch % 10 == 0:
                saver.save(sess, config.ckpt, global_step=pre_epoch)
        
        generate_negative_samples(sess, gen_dataloader, dst_path=config.negative_file)

        train_discriminator(sess, discriminator, epoch_num=config.discriminator_pretrain_epoch)

        opt_vars['learning_rate'] = config.update_init_lr if config.update_init_lr > opt_vars['learning_rate'] else opt_vars['learning_rate']

        for update_epoch in range(1, config.adversial_epoch + 1):
            train_ppl = update_generator(sess, gen_dataloader)
            generate_negative_samples(sess, gen_dataloader, dst_path=config.negative_file)
            train_discriminator(sess, discriminator, epoch_num=1)
            if update_epoch % 10 == 0:
                saver.save(sess, config.ckpt, global_step=update_epoch + config.generator_pretrain_epoch)
