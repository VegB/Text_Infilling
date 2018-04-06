import tensorflow as tf
import importlib
from shutil import copyfile

import texar as tx

from generator import Generator
from discriminator import Discriminator
from dataloader import GenDataLoader, DisDataLoader
from rollout import Rollout
from utils import print_result, store_output, pad_to_length


config_path = "config"
config = importlib.import_module(config_path)


def pretrain_generator(sess, generator, input_file, vocab_file, epoch_num=1):
    dataloader = GenDataLoader(config, text_file=input_file,
                               vocab_file=vocab_file, epoch_num=epoch_num)

    while not dataloader.should_stop():
        _, step, loss, outputs = sess.run([generator.train_op, generator.global_step,
                                           generator.mle_loss, generator.outputs],
                                          feed_dict={generator.data_batch: dataloader.get_batch(),
                                                     generator.global_step: dataloader.step,
                                                     tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        if step % 10 == 0:
            print("%d: %.6f" % (step, loss))
            print_result(outputs.sample_id, dataloader.id2word, dataloader.max_len)


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
        ids, labels = dataloader.get_batch()

        _, step, loss = sess.run([discriminator.train_op, discriminator.global_step,
                                  discriminator.mle_loss],
                                 feed_dict={discriminator.samples: ids,
                                            discriminator.labels: labels,
                                            discriminator.global_step: dataloader.step,
                                            tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        if step % 10 == 0:
            print("%d: %.6f" % (step, loss))


def update_generator_with_rollout(sess, generator, discriminator, update_step=1):
    dataloader = GenDataLoader(config, text_file=config.train_file,
                               vocab_file=config.vocab_file, epoch_num=1)
    rollout = Rollout(config, generator, update_rate=0.8)

    for step in range(update_step):
        print("step: ", step)
        decode_output = sess.run(generator.generated_outputs,
                                 feed_dict={generator.data_batch: dataloader.get_batch(),
                                            tx.global_mode(): tf.estimator.ModeKeys.EVAL})
        generated_samples = [pad_to_length(content, max_len=config.num_steps,
                                           bos=dataloader.bos_id,
                                           eos=dataloader.eos_id, pad=dataloader.pad_id)
                             for content in decode_output.sample_id]  # [batch_size, max_len + 2]
        rewards = rollout.get_reward(sess, generated_samples=generated_samples,
                                     rollout_num=config.rollout_num, discriminator=discriminator)
        print("rewards:\n", rewards)
        _ = sess.run(generator.update_op, feed_dict={generator.data_batch: generated_samples,
                                                     generator.rewards: rewards,
                                                     tx.global_mode(): tf.estimator.ModeKeys.TRAIN})


if __name__ == "__main__":
    dataloader = GenDataLoader(config, text_file=config.train_file,
                               vocab_file=config.vocab_file, epoch_num=1)
    generator = Generator(config, word2id=dataloader.word2id, bos=dataloader.bos_id,
                          eos=dataloader.eos_id, pad=dataloader.pad_id)
    discriminator = Discriminator(config, word2id=dataloader.word2id)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        for i in range(2):
            pretrain_generator(sess, generator, config.train_file, config.vocab_file,
                               epoch_num=int(config.generator_pretrain_epoch / 10))
            train_rst_file = "./data/%d.txt" % (i * 10)
            eval_rst_file = "./data/%d.eval.txt" % (i * 10)
            generate_negative_samples(sess, generator, input_file=config.train_file,
                                      vocab_file=config.vocab_file, dst_path=train_rst_file)
            generate_negative_samples(sess, generator, input_file=config.eval_file,
                                      vocab_file=config.vocab_file, dst_path=eval_rst_file)

        train_discriminator(sess, discriminator, positive_file=config.train_file,
                            negative_file=config.negative_file, vocab_file=config.vocab_file,
                            epoch_num=config.discriminator_pretrain_epoch)

        for update_epoch in range(config.adversial_epoch):
            update_generator_with_rollout(sess, generator, discriminator, update_step=1)
            generate_negative_samples(sess, generator, input_file=config.train_file,
                                      vocab_file=config.vocab_file, dst_path=config.negative_file)
            train_discriminator(sess, discriminator, positive_file=config.train_file,
                                negative_file=config.negative_file, vocab_file=config.vocab_file,
                                epoch_num=15)
            if update_epoch % 10 == 0 or update_epoch == config.adversial_epoch - 1:
                print(update_epoch)
                train_rst_file = "./data/%d.txt" % (update_epoch + config.generator_pretrain_epoch)
                eval_rst_file = "./data/%d.eval.txt" % (update_epoch + config.generator_pretrain_epoch)
                copyfile(config.negative_file, train_rst_file)
                generate_negative_samples(sess, generator, input_file=config.eval_file,
                                          vocab_file=config.vocab_file, dst_path=eval_rst_file)
