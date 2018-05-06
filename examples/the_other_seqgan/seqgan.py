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


def pretrain_generator(sess, generator, input_file, vocab_file, epoch_num=1):
    print("-------------Pretrain Generator----------------")
    dataloader = GenDataLoader(config, text_file=input_file,
                               vocab_file=vocab_file, epoch_num=epoch_num)

    while not dataloader.should_stop():
        _, step, loss, outputs = sess.run([generator.train_op, generator.global_step,
                                           generator.teacher_loss, generator.outputs],
                                          feed_dict={generator.data_batch: dataloader.get_batch(),
                                                     generator.global_step: dataloader.step,
                                                     tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        if step % 200 == 0:
            print("%d: teacher_loss = %.6f" % (step, loss))
            print_result(outputs.sample_id[:config.print_num], dataloader.id2word, dataloader.max_len)


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
    print_result(generated_outputs[:config.print_num], dataloader.id2word, dataloader.max_len)


def train_discriminator(sess, discriminator, positive_file, negative_file, vocab_file, epoch_num):
    print("-------------Train Discriminator----------------")
    dataloader = DisDataLoader(config, epoch_num=epoch_num, positive_file=positive_file,
                               negative_file=negative_file, vocab_file=vocab_file)

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


def update_generator(sess, generator, discriminator, positive_file, negative_file, vocab_file):
    print("-------------Update Generator----------------")
    dataloader = DisDataLoader(config, epoch_num=1, positive_file=positive_file,
                               negative_file=negative_file, vocab_file=vocab_file)

    for i in range(config.g_update_batch):
        gen_data = sess.run(generator.generated_outputs,
                            feed_dict={tx.global_mode(): tf.estimator.ModeKeys.EVAL})
        g_data = [pad_to_length(sent, bos=dataloader.bos_id, eos=dataloader.eos_id,
                                pad=dataloader.pad_id, max_len=dataloader.max_len)
                                for sent in gen_data.sample_id]

        r_ids, _ = dataloader.get_batch()

        _, g_preds = sess.run([discriminator.r_preds, discriminator.g_preds],
                              feed_dict={discriminator.real_samples: r_ids,
                                         discriminator.gen_samples: [line[1:] for line in g_data],
                                         discriminator.global_step: dataloader.step,
                                         tx.global_mode(): tf.estimator.ModeKeys.TRAIN})

        _, update_loss = sess.run([generator.update_op, generator.update_loss],
                                  feed_dict={generator.data_batch: g_data,
                                             generator.rewards: [preds[:-1] for preds in g_preds],
                                             tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        print("%d: update_total_loss = %.6f" % (i, update_loss))


def calculate_nll(sess, generator, oracle_file, gen_file, vocab_file, epoch_id, mode):
    # NLL Oracle
    dataloader = GenDataLoader(config, text_file=oracle_file,
                               vocab_file=vocab_file, epoch_num=1)
    nll = []
    for i in range(50):
        loss = sess.run([generator.teacher_loss],
                        feed_dict={generator.data_batch: dataloader.get_batch(),
                                   tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        nll.append(loss)
    nll_oracle = np.mean(nll)
    print("%s epoch %d: nll_oracle = %f" % (mode, epoch_id, nll_oracle))
    log.write("%s epoch %d: nll_oracle = %f\n" % (mode, epoch_id, nll_oracle))

    # NLL Gen
    dataloader = GenDataLoader(config, text_file=gen_file,
                               vocab_file=vocab_file, epoch_num=1)
    nll = []
    for i in range(50):
        loss = sess.run([generator.teacher_loss],
                        feed_dict={generator.data_batch: dataloader.get_batch(),
                                   tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        nll.append(loss)
    nll_gen = np.mean(nll)
    print("%s epoch %d: nll_gen = %f" % (mode, epoch_id, nll_gen))
    log.write("%s epoch %d: nll_gen = %f\n" % (mode, epoch_id, nll_gen))


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

        generate_negative_samples(sess, generator, config.train_file, config.vocab_file,
                                  dst_path="./data/0.txt")

        for pre_epoch in range(1, config.generator_pretrain_epoch + 1):
            pretrain_generator(sess, generator, config.train_file, config.vocab_file)
            generate_negative_samples(sess, generator, config.train_file, config.vocab_file,
                                      dst_path=config.negative_file)
            calculate_nll(sess, generator, epoch_id=pre_epoch, oracle_file=config.train_file,
                          gen_file=config.negative_file, vocab_file=config.vocab_file,
                          mode="Pretrain")
            if pre_epoch % 10 == 0:
                train_rst_file = "./data/%d.txt" % pre_epoch
                copyfile(config.negative_file, train_rst_file)
                saver.save(sess, config.ckpt, global_step=pre_epoch)

        train_discriminator(sess, discriminator, positive_file=config.train_file,
                            negative_file=config.negative_file, vocab_file=config.vocab_file,
                            epoch_num=config.discriminator_pretrain_epoch)
        
        for update_epoch in range(1, config.adversial_epoch + 1):
            update_generator(sess, generator, discriminator, positive_file=config.train_file,
                             negative_file=config.negative_file, vocab_file=config.vocab_file)
            generate_negative_samples(sess, generator, input_file=config.train_file,
                                      vocab_file=config.vocab_file, dst_path=config.negative_file)
            calculate_nll(sess, generator, epoch_id=update_epoch, oracle_file=config.train_file,
                          gen_file=config.negative_file, vocab_file=config.vocab_file,
                          mode="Adversarial")
            train_discriminator(sess, discriminator, positive_file=config.train_file,
                                negative_file=config.negative_file, vocab_file=config.vocab_file,
                                epoch_num=1)
            if update_epoch % 10 == 0:
                current_epoch = update_epoch + config.generator_pretrain_epoch
                train_rst_file = "./data/%d.txt" % current_epoch
                copyfile(config.negative_file, train_rst_file)
                saver.save(sess, config.ckpt, global_step=current_epoch)
