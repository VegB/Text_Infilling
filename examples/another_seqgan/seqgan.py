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


def generate_negative_samples(sess, generator, gen_dataloader, dst_path):
    print("-------------Generate Negative Samples----------------")
    gen_dataloader.reset()
    generated_outputs = []
    while not gen_dataloader.should_stop():
        decode_output = sess.run(generator.generated_sample_id,
                                 feed_dict={generator.data_batch: gen_dataloader.get_batch(),
                                            tx.global_mode(): tf.estimator.ModeKeys.EVAL})
        generated_outputs.extend(decode_output)

    store_output(output=generated_outputs, id2word=gen_dataloader.id2word,
                 data_path=dst_path, max_len=gen_dataloader.max_len)
    print_result(generated_outputs[:config.print_num], gen_dataloader.id2word, gen_dataloader.max_len)


def train_discriminator(sess, discriminator, epoch_num):
    print("-------------Train Discriminator----------------")
    dataloader = DisDataLoader(config, epoch_num=epoch_num, positive_file=config.train_file,
                               negative_file=config.negative_file, vocab_file=config.vocab_file)

    while not dataloader.should_stop():
        r_ids, g_ids = dataloader.get_batch()

        _, step, loss = sess.run([discriminator.train_op, discriminator.global_step,
                                  discriminator.total_loss],
                                 feed_dict={discriminator.real_samples: r_ids,
                                            discriminator.gen_samples: g_ids,
                                            discriminator.global_step: dataloader.step,
                                            tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        if step % 200 == 0:
            print("%d: dis_total_loss: %.6f" % (step, loss))


def update_generator(sess, generator, discriminator, gen_dataloader, dis_dataloader):
    print("-------------Update Generator----------------")
    loss = 0.
    iters = 0
    for i in range(config.g_update_batch):
        if gen_dataloader.should_stop():
            gen_dataloader.reset()
        if dis_dataloader.should_stop():
            dis_dataloader.reset()
        # Teacher forcing
        _, mle_loss, step = sess.run([generator.train_op, generator.teacher_loss, generator.global_step],
                                   feed_dict={generator.data_batch: gen_dataloader.get_batch(),
                                              generator.learning_rate: opt_vars['learning_rate'],
                                              tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        # calculate current ppl
        loss += mle_loss
        iters += config.num_steps
        ppl = np.exp(loss / iters)
        print('global step:', step, ' ' * 4, 'training ppl:', ppl)

        gen_data = sess.run(generator.generated_sample_id,
                            feed_dict={tx.global_mode(): tf.estimator.ModeKeys.EVAL})
        g_data = [pad_to_length(sent, bos=dis_dataloader.bos_id, eos=dis_dataloader.eos_id,
                                pad=dis_dataloader.pad_id, max_len=dis_dataloader.max_len)
                  for sent in gen_data]

        r_ids, _ = dis_dataloader.get_batch()

        _, g_preds = sess.run([discriminator.r_preds, discriminator.g_preds],
                              feed_dict={discriminator.real_samples: r_ids,
                                         discriminator.gen_samples: [line[1:] for line in g_data],
                                         discriminator.global_step: dis_dataloader.step,
                                         tx.global_mode(): tf.estimator.ModeKeys.TRAIN})

        _, update_loss = sess.run([generator.update_op, generator.update_loss],
                                  feed_dict={generator.data_batch: g_data,
                                             generator.rewards: [preds[:-1] for preds in g_preds],
                                             generator.learning_rate: opt_vars['learning_rate'],
                                             tx.global_mode(): tf.estimator.ModeKeys.TRAIN})

    return np.exp(loss / iters)


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
    return ppl


def record_ppl(sess, generator, valid_dataloader, test_dataloader, epoch_id, train_ppl, mode="Pretrain"):
    valid_ppl = calculate_ppl(sess, generator, valid_dataloader)
    test_ppl = calculate_ppl(sess, generator, test_dataloader)
    rst = "epoch %d(%s): learning_rate = %.10f, train_ppl = %f, valid_ppl = %f, test_ppl = %f\n" % \
          (epoch_id, mode, opt_vars["learning_rate"], train_ppl, valid_ppl, test_ppl)
    print_and_write_to_file(rst, log)


if __name__ == "__main__":
    word2id = tx.data.make_vocab(config.train_file, newline_token="<EOS>", return_type="dict")
    gen_dataloader = GenDataLoader(config, config.train_file, word2id)
    valid_dataloader = GenDataLoader(config, config.valid_file, word2id)
    test_dataloader = GenDataLoader(config, config.test_file, word2id)
    # dis_dataloader = DisDataLoader(config, epoch_num=1, positive_file=config.train_file,
    #                                negative_file=config.train_file, vocab_file=config.vocab_file)

    generator = Generator(vocab_size=len(word2id))
    # discriminator = Discriminator(config, word2id=dis_dataloader.word2id)
    saver = tf.train.Saver()

    # ------------Pretrain---------------
    batch_size = tf.placeholder(dtype=tf.int32, shape=())
    num_steps = config.num_steps

    opt_vars = {
        'learning_rate': 0.003,
        'best_valid_ppl': 1e100,
        'steps_not_improved': 0
    }

    inputs = tf.placeholder(tf.int32, [None, num_steps])
    targets = tf.placeholder(tf.int32, [None, num_steps])

    initial_state, logits, final_state = \
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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        for pre_epoch in range(1, config.generator_pretrain_epoch + 1):
            train_ppl = pretrain_generator(sess, gen_dataloader, valid_dataloader, test_dataloader)
            if pre_epoch % 10 == 0:
                saver.save(sess, config.ckpt, global_step=pre_epoch)

        ''' 
        generate_negative_samples(sess, generator, gen_dataloader, dst_path=config.negative_file)
        
        train_discriminator(sess, discriminator, epoch_num=config.discriminator_pretrain_epoch)
        saver.save(sess, './checkpoint/pretrained/ckpt', global_step=80)

        for update_epoch in range(1, config.adversial_epoch + 1):
            train_ppl = update_generator(sess, generator, discriminator, gen_dataloader, dis_dataloader)
            generate_negative_samples(sess, generator, gen_dataloader, dst_path=config.negative_file)
            record_ppl(sess, generator, valid_dataloader, test_dataloader, epoch_id=update_epoch,
                       train_ppl=train_ppl, mode="Adversarial")
            train_discriminator(sess, discriminator, epoch_num=1)
            if update_epoch % 10 == 0:
                saver.save(sess, config.ckpt, global_step=update_epoch + config.generator_pretrain_epoch)
        '''
