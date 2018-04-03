import tensorflow as tf
import numpy as np

from txtgen.modules import get_helper
from txtgen import context

from utils import *


class Rollout:
    def __init__(self, generator, update_rate):
        with tf.variable_scope('rollout'):
            self.generator = generator
            self.update_rate = update_rate
            self.batch_size = self.generator.batch_size
            self.max_seq_length = self.generator.max_seq_length

            self.given_num = tf.placeholder(dtype=tf.int32, shape=(), name="given_num")
            self.data_batch = tf.placeholder(dtype=tf.int32, name="data_batch",
                                             shape=(self.batch_size, self.max_seq_length + 2))
            self.bos_id = self.generator.bos_id
            self.eos_id = self.generator.eos_id

            # builder encoder
            self.encoder = self.generator.encoder

            # Build decoder. Simply use the default hyperparameters.
            self.decoder = self.generator.decoder

            # Build connector
            self.connector = self.generator.connector

            enc_outputs, enc_last = self.encoder(inputs=self.data_batch[:, tf.newaxis])

            # When current index i < given_num, use the provided tokens as the input at each time step
            helper_train = get_helper(
                self.decoder.hparams.helper_train.type,
                inputs=self.data_batch[:, :self.given_num],
                sequence_length=[self.given_num] * self.batch_size,
                embedding=self.decoder.embedding)

            _, final_state, sequence_lengths = self.decoder(
                helper=helper_train, initial_state=self.connector(enc_last))

            # current index i >= given_num, start roll-out, use the output at time step t as the input at time step t+1
            helper_infer = get_helper(
                self.decoder.hparams.helper_infer.type,
                embedding=self.decoder.embedding,
                start_tokens=[self.bos_id] * self.batch_size,
                end_token=self.eos_id,
                softmax_temperature=None,
                seed=None)

            final_outputs, final_state, sequence_lengths = self.decoder(
                helper=helper_infer, initial_state=self.connector(final_state))

            self.result = tf.concat([self.data_batch[:, 1:self.given_num + 1],
                                     final_outputs.sample_id[:, :self.max_seq_length - self.given_num + 2]], 1)  # [batch, self.max_len + 2]

    def get_reward(self, sess, generated_samples, rollout_num, discriminator):
        rewards = []
        for i in range(rollout_num):
            for given_num in range(1, self.max_seq_length):
                samples = sess.run(self.result, feed_dict={self.data_batch: generated_samples,
                                                           self.given_num:given_num,
                                                           context.is_train(): False})
                samples = [pad_to_length(content, max_len=self.max_seq_length, eos=self.eos_id,
                                         pad=self.generator.pad_id) for content in samples]  # [batch_size, max_len + 1]
                ypred_for_auc = sess.run(discriminator.ypred_for_auc,
                                         feed_dict={discriminator.samples: samples,
                                                    context.is_train(): False})
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            print(type(generated_samples))
            ypred_for_auc = sess.run(discriminator.ypred_for_auc,
                                     feed_dict={discriminator.samples: [sample[:self.max_seq_length+1] for sample in generated_samples],
                                                context.is_train(): False})
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.max_seq_length - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards
