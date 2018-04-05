import tensorflow as tf
import importlib

import texar as tx

from generator import Generator
from discriminator import Discriminator
from dataloader import GenDataLoader, DisDataLoader
from rollout import Rollout
from utils import *


train_file = "../../data/small_sent.txt"
vocab_file = "../../data/vocab.txt"
# train_file = "../../data/coco.txt"
# vocab_file = "../../data/coco_vocab.txt"
positive_file = "./data/positive.txt"
negative_file = "./data/negative.txt"
config_path = "config"

config = importlib.import_module(config_path)

ROLLOUT_NUM = 1
PRE_EPOCH_NUM_T = 1
PRE_EPOCH_NUM_G = 40
PRE_EPOCH_NUM_D = 50
ADVER_BATCH = 20


def pretrain_generator(sess, generator, train_file, vocab_file, epoch_num=1):
    dataloader = GenDataLoader(config, text_file=train_file,
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


def generate_negative_samples(sess, generator, train_file, vocab_file, dst_path):
    dataloader = GenDataLoader(config, text_file=train_file,
                               vocab_file=vocab_file, epoch_num=1)

    generated_outputs = []
    while not dataloader.should_stop():
        decode_output = sess.run(generator.generated_outputs,
                                 feed_dict={generator.data_batch: dataloader.get_batch(),
                                            tx.global_mode(): tf.estimator.ModeKeys.EVAL})
        generated_outputs.extend(decode_output.sample_id)

    store_output(output=generated_outputs, id2word=dataloader.id2word,
                 data_path=dst_path, max_len=dataloader.max_len)


def pretrain_discriminator(sess, discriminator, positive_file, negative_file, vocab_file, epoch_num):
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
    dataloader = GenDataLoader(config, text_file=train_file,
                               vocab_file=vocab_file, epoch_num=1)
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
                                     rollout_num=ROLLOUT_NUM, discriminator=discriminator)
        print("rewards:\n", rewards)
        _ = sess.run(generator.update_op, feed_dict={generator.data_batch: generated_samples,
                                                     generator.rewards: rewards,
                                                     tx.global_mode(): tf.estimator.ModeKeys.TRAIN})


if __name__ == "__main__":
    dataloader = GenDataLoader(config, text_file=train_file, vocab_file=vocab_file, epoch_num=1)
    generator = Generator(config, word2id=dataloader.word2id, bos=dataloader.bos_id,
                          eos=dataloader.eos_id, pad=dataloader.pad_id)
    discriminator = Discriminator(config, word2id=dataloader.word2id)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        pretrain_generator(sess, generator, train_file, vocab_file, epoch_num=3)
        generate_negative_samples(sess, generator, train_file=train_file, vocab_file=vocab_file, dst_path=negative_file)

        pretrain_discriminator(sess, discriminator, positive_file=train_file,
                               negative_file=negative_file, vocab_file=vocab_file, epoch_num=5)

        for batch_cnt in range(ADVER_BATCH):
            update_generator_with_rollout(sess, generator, discriminator, update_step=2)
            generate_negative_samples(sess, generator, train_file=train_file, vocab_file=vocab_file,
                                      dst_path=negative_file)
            pretrain_discriminator(sess, discriminator, positive_file=train_file,
                                   negative_file=negative_file, vocab_file=vocab_file, epoch_num=5)
