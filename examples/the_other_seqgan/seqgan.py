import tensorflow as tf
import numpy as np
import importlib
from shutil import copyfile

import texar as tx

from generator import Generator
from discriminator import Discriminator
from dataloader import GenDataLoader, DisDataLoader
from utils import print_result, store_output, pad_to_length


config_path = "config"
config = importlib.import_module(config_path)
log = open(config.log_file, "w")


def pretrain_generator(sess, generator, input_file, vocab_file, epoch_id, epoch_num=1):
    print("-------------Pretrain Generator----------------")
    dataloader = GenDataLoader(config, text_file=input_file,
                               vocab_file=vocab_file, epoch_num=epoch_num)

    nll = []
    while not dataloader.should_stop():
        _, step, loss, outputs = sess.run([generator.train_op, generator.global_step,
                                           generator.teacher_loss, generator.outputs],
                                          feed_dict={generator.data_batch: dataloader.get_batch(),
                                                     generator.global_step: dataloader.step,
                                                     tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        nll.append(loss)
        if step % 200 == 0:
            print("%d: %.6f" % (step, loss))
            print_result(outputs.sample_id, dataloader.id2word, dataloader.max_len)
    nll_test = np.mean(nll)
    print("Pretrain epoch %d: nll_test = %f" % (epoch_id*epoch_num, nll_test))
    log.write("Pretrain epoch %d: nll_test = %f\n" % (epoch_id*epoch_num, nll_test))


def generate_negative_samples(sess, generator, input_file, vocab_file, dst_path):
    print("-------------Generate Negative Samples----------------")
    dataloader = GenDataLoader(config, text_file=input_file,
                               vocab_file=vocab_file, epoch_num=1)

    generated_outputs = []
    while not dataloader.should_stop():
        decode_output = sess.run(generator.generated_outputs,
                                 feed_dict={generator.data_batch: dataloader.get_batch(),
                                            tx.global_mode(): tf.estimator.ModeKeys.EVAL})
        generated_outputs.extend(decode_output.sample_id)

    store_output(output=generated_outputs, id2word=dataloader.id2word,
                 data_path=dst_path, max_len=dataloader.max_len)
    print_result(generated_outputs[:10], dataloader.id2word, dataloader.max_len)


def train_discriminator(sess, discriminator, positive_file, negative_file, vocab_file, epoch_num):
    print("-------------Train Discriminator----------------")
    dataloader = DisDataLoader(config, epoch_num=epoch_num, positive_file=positive_file,
                               negative_file=negative_file, vocab_file=vocab_file)

    while not dataloader.should_stop():
        r_ids, g_ids = dataloader.get_batch()

        _, step, loss = sess.run([discriminator.train_op, discriminator.global_step,
                                  discriminator.dis_loss],
                                 feed_dict={discriminator.real_samples: r_ids,
                                            discriminator.gen_samples: g_ids,
                                            discriminator.global_step: dataloader.step,
                                            tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        if step % 20 == 0:
            print("%d: %.6f" % (step, loss))


def update_generator(sess, generator, discriminator, positive_file, negative_file, vocab_file,
                     epoch_num):
    print("-------------Update Generator----------------")
    dataloader = DisDataLoader(config, epoch_num=epoch_num, positive_file=positive_file,
                               negative_file=negative_file, vocab_file=vocab_file)

    while not dataloader.should_stop():
        gen_data = sess.run(generator.generated_outputs,
                            feed_dict={tx.global_mode(): tf.estimator.ModeKeys.EVAL})
        g_data = [pad_to_length(sent, eos=dataloader.eos_id, pad=dataloader.pad_id,
                                max_len=dataloader.max_len) for sent in gen_data.sample_id]

        r_ids, _ = dataloader.get_batch()

        _, g_preds = sess.run([discriminator.r_preds, discriminator.g_preds],
                              feed_dict={discriminator.real_samples: r_ids,
                                         discriminator.gen_samples: g_data,
                                         discriminator.global_step: dataloader.step,
                                         tx.global_mode(): tf.estimator.ModeKeys.TRAIN})

        _, _, step, update_loss = sess.run([generator.exp_op, generator.update_op,
                                            generator.update_step, generator.gen_loss],
                                           feed_dict={generator.rewards: g_preds[:, :-1, tf.newaxis],
                                                      generator.update_step: dataloader.step,
                                                      tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        if step % 5:
            print("%d: %.6f" % (step, update_loss))


def calculate_nll(sess, generator, input_file, vocab_file, epoch_id):
    dataloader = GenDataLoader(config, text_file=input_file,
                               vocab_file=vocab_file, epoch_num=1)
    nll = []
    while not dataloader.should_stop():
        loss = sess.run([generator.teacher_loss],
                        feed_dict={generator.data_batch: dataloader.get_batch(),
                                   tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        nll.append(loss)
    nll_oracle = np.mean(nll)
    print("Adversial epoch %d: nll_test = %f" % (epoch_id, nll_oracle))
    log.write("Adversial epoch %d: nll_test = %f\n" % (epoch_id, nll_oracle))


if __name__ == "__main__":
    dataloader = GenDataLoader(config, text_file=config.train_file,
                               vocab_file=config.vocab_file, epoch_num=1)
    generator = Generator(config, word2id=dataloader.word2id, bos=dataloader.bos_id,
                          eos=dataloader.eos_id, pad=dataloader.pad_id)
    discriminator = Discriminator(config, word2id=dataloader.word2id)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        for i in range(8):
            pretrain_generator(sess, generator, config.train_file, config.vocab_file,
                               epoch_id=i, epoch_num=int(config.generator_pretrain_epoch / 8))
            train_rst_file = "./data/%d.txt" % (i * 10)
            generate_negative_samples(sess, generator, input_file=config.train_file,
                                      vocab_file=config.vocab_file, dst_path=train_rst_file)
            if i > 0 and i % 2 == 0:
                saver.save(sess, config.ckpt, global_step=i * 10)

        generate_negative_samples(sess, generator, input_file=config.train_file,
                                  vocab_file=config.vocab_file, dst_path=config.negative_file)

        train_discriminator(sess, discriminator, positive_file=config.train_file,
                            negative_file=config.negative_file, vocab_file=config.vocab_file,
                            epoch_num=config.discriminator_pretrain_epoch)

        for update_epoch in range(config.adversial_epoch):
            update_generator(sess, generator, discriminator, positive_file=config.train_file,
                             negative_file=config.negative_file, vocab_file=config.vocab_file,
                             epoch_num=1)
            generate_negative_samples(sess, generator, input_file=config.train_file,
                                      vocab_file=config.vocab_file, dst_path=config.negative_file)
            calculate_nll(sess, generator, input_file=config.negative_file,
                          vocab_file=config.vocab_file, epoch_id=update_epoch)
            train_discriminator(sess, discriminator, positive_file=config.train_file,
                                negative_file=config.negative_file, vocab_file=config.vocab_file,
                                epoch_num=1)
            if update_epoch % 10 == 0 or update_epoch == config.adversial_epoch - 1:
                current_epoch = update_epoch + config.generator_pretrain_epoch
                train_rst_file = "./data/%d.txt" % current_epoch
                copyfile(config.negative_file, train_rst_file)
                saver.save(sess, config.ckpt, global_step=current_epoch)
