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


def train_discriminator(sess, discriminator, positive_file, negative_file, vocab_file, epoch_num):
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

        # for i in range(8):
        #     pretrain_generator(sess, generator, config.train_file, config.vocab_file,
        #                        epoch_id=i, epoch_num=int(config.generator_pretrain_epoch / 8))
        #     train_rst_file = "./data/%d.txt" % (i * 10)
        #     eval_rst_file = "./data/%d.eval.txt" % (i * 10)
        #     generate_negative_samples(sess, generator, input_file=config.train_file,
        #                               vocab_file=config.vocab_file, dst_path=train_rst_file)
        #     generate_negative_samples(sess, generator, input_file=config.eval_file,
        #                               vocab_file=config.vocab_file, dst_path=eval_rst_file)
        #     if i > 0 and i % 2 == 0:
        #         saver.save(sess, config.ckpt, global_step=i*10)
        #
        # generate_negative_samples(sess, generator, input_file=config.train_file, vocab_file=config.vocab_file, dst_path=config.negative_file)
        #
        train_discriminator(sess, discriminator, positive_file=config.train_file,
                            negative_file=config.negative_file, vocab_file=config.vocab_file,
                            epoch_num=config.discriminator_pretrain_epoch)
        #
        # for update_epoch in range(config.adversial_epoch):
        #     update_generator_with_rollout(sess, generator, discriminator, update_step=1)
        #     generate_negative_samples(sess, generator, input_file=config.train_file,
        #                               vocab_file=config.vocab_file, dst_path=config.negative_file)
        #     calculate_nll(sess, generator, input_file=config.negative_file,
        #                   vocab_file=config.vocab_file, epoch_id=update_epoch)
        #     train_discriminator(sess, discriminator, positive_file=config.train_file,
        #                         negative_file=config.negative_file, vocab_file=config.vocab_file,
        #                         epoch_num=15)
        #     if update_epoch % 10 == 0 or update_epoch == config.adversial_epoch - 1:
        #         print(update_epoch)
        #         train_rst_file = "./data/%d.txt" % (update_epoch + config.generator_pretrain_epoch)
        #         eval_rst_file = "./data/%d.eval.txt" % (update_epoch + config.generator_pretrain_epoch)
        #         copyfile(config.negative_file, train_rst_file)
        #         generate_negative_samples(sess, generator, input_file=config.eval_file,
        #                                   vocab_file=config.vocab_file, dst_path=eval_rst_file)
        #         saver.save(sess, config.ckpt, global_step=config.generator_pretrain_epoch + update_epoch)
