import tensorflow as tf
import numpy as np
import importlib
from shutil import copyfile

import texar as tx

from oracle import OracleLSTM
from generator import Generator
from discriminator import Discriminator
from dataloader import GenDataLoader, DisDataLoader
from rollout import Rollout
from utils import print_result, store_output, pad_to_length


config_path = "config_synthetic"
config = importlib.import_module(config_path)
log = open(config.log_file, "w")


def pretrain_generator(sess, generator, input_file, vocab_file, epoch_id, epoch_num=1):
    dataloader = GenDataLoader(config, text_file=input_file,
                               vocab_file=vocab_file, epoch_num=epoch_num)
    nll = []
    while not dataloader.should_stop():
        _, step, loss, outputs = sess.run([generator.train_op, generator.global_step,
                                           generator.mle_loss, generator.outputs],
                                          feed_dict={generator.data_batch: dataloader.get_batch(),
                                                     generator.global_step: dataloader.step,
                                                     tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        nll.append(loss)
        if step % 200 == 0:
            print("%d: %.6f" % (step, loss))
            print_result(outputs.sample_id, dataloader.id2word, dataloader.max_len)

    nll_test = np.mean(nll)
    print("Pretrain epoch %d: nll_test = %f" % (epoch_id, nll_test))
    log.write("Pretrain epoch %d: nll_test = %f\n" % (epoch_id, nll_test))


def generate_samples(sess, generator, input_file, vocab_file, dst_path):
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
        if step % 200 == 0:
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


def calculate_nll(sess, target_generator, input_file, vocab_file, epoch_id):
    dataloader = GenDataLoader(config, text_file=input_file,
                               vocab_file=vocab_file, epoch_num=1)
    nll = []
    while not dataloader.should_stop():
        loss = sess.run([target_generator.mle_loss],
                        feed_dict={target_generator.data_batch: dataloader.get_batch(),
                                   tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        nll.append(loss)
    nll_oracle = np.mean(nll)
    print("epoch %d: nll_oracle = %f" % (epoch_id, nll_oracle))
    log.write("epoch %d: nll_oracle = %f\n" % (epoch_id, nll_oracle))


if __name__ == "__main__":
    dataloader = GenDataLoader(config, text_file=config.train_file,
                               vocab_file=config.vocab_file, epoch_num=1)
    target_generator = OracleLSTM(config, word2id=dataloader.word2id, bos=dataloader.bos_id,
                                  eos=dataloader.eos_id, pad=dataloader.pad_id)
    generator = Generator(config, word2id=dataloader.word2id, bos=dataloader.bos_id,
                          eos=dataloader.eos_id, pad=dataloader.pad_id)
    discriminator = Discriminator(config, word2id=dataloader.word2id)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # Pretrain TargetGenerator
        generate_samples(sess, target_generator, input_file=config.train_file,
                         vocab_file=config.vocab_file, dst_path=config.positive_file)

        # Pretrain Generator
        for train_epoch in range(config.generator_pretrain_epoch):
            pretrain_generator(sess, generator, config.positive_file, config.vocab_file,
                               epoch_id=train_epoch, epoch_num=1)
            generate_samples(sess, generator, input_file=config.train_file,
                             vocab_file=config.vocab_file, dst_path=config.negative_file)
            calculate_nll(sess, target_generator, input_file=config.negative_file,
                          vocab_file=config.vocab_file, epoch_id=train_epoch)

        # Pretrain Discriminator
        train_discriminator(sess, discriminator, positive_file=config.positive_file,
                            negative_file=config.negative_file, vocab_file=config.vocab_file,
                            epoch_num=config.discriminator_pretrain_epoch)

        # Adversial Training
        for adv_epoch in range(config.adversial_epoch):
            update_generator_with_rollout(sess, generator, discriminator,
                                          update_step=config.adv_g_step)
            generate_samples(sess, generator, input_file=config.train_file,
                             vocab_file=config.vocab_file, dst_path=config.negative_file)
            calculate_nll(sess, target_generator, input_file=config.negative_file,
                          vocab_file=config.vocab_file, epoch_id=adv_epoch)
            train_discriminator(sess, discriminator, positive_file=config.train_file,
                                negative_file=config.negative_file, vocab_file=config.vocab_file,
                                epoch_num=config.adv_d_epoch)

    log.close()
