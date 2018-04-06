import tensorflow as tf
import numpy as np

import texar as tx

from utils import *


class Rollout:
    def __init__(self, config, generator, update_rate):
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

            self.embedder = self.generator.embedder
            self.encoder = self.generator.encoder
            self.decoder = self.generator.decoder
            self.connector = self.generator.connector

            emb_inputs = self.embedder(self.data_batch[:, :-1])
            if config.keep_prob < 1:
                emb_inputs = tf.nn.dropout(
                    emb_inputs, tx.utils.switch_dropout(config.keep_prob))

            enc_outputs, enc_last = self.encoder(inputs=emb_inputs)

            # When current index i < given_num,
            # use the provided tokens as the input at each time step
            self.outputs, final_state, seq_lengths = self.decoder(
                decoding_strategy="train_greedy",
                impute_finished=True,
                inputs=self.data_batch[:, :self.given_num],
                sequence_length=[self.given_num] * self.batch_size,
                embedding=self.embedder,
                initial_state=self.connector(enc_last))

            # current index i >= given_num, start roll-out,
            # use the output at time step t as the input at time step t+1
            final_outputs, _, _ = self.decoder(
                decoding_strategy="infer_sample",
                start_tokens=self.data_batch[:, self.given_num - 1],
                end_token=self.eos_id,
                embedding=self.embedder,
                initial_state=self.connector(final_state))

            self.result = tf.concat([self.data_batch[:, 1:self.given_num + 1],
                                     final_outputs.sample_id[:,
                                     :self.max_seq_length - self.given_num + 2]], 1)  # [batch, self.max_len + 2]

    def get_reward(self, sess, generated_samples, rollout_num, discriminator):
        rewards = []
        for i in range(rollout_num):
            for given_num in range(1, self.max_seq_length):
                samples = sess.run(self.result, feed_dict={self.data_batch: generated_samples,
                                                           self.given_num: given_num,
                                                           tx.global_mode(): tf.estimator.ModeKeys.EVAL})
                samples = [pad_to_length(content, max_len=self.max_seq_length, eos=self.eos_id,
                                         pad=self.generator.pad_id) for content in samples]  # [batch_size, max_len + 1]
                ypred_for_auc = sess.run(discriminator.ypred_for_auc,
                                         feed_dict={discriminator.samples: samples,
                                                    tx.global_mode(): tf.estimator.ModeKeys.EVAL})
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed_dict_ = {discriminator.samples:
                          [sample[:self.max_seq_length+1] for sample in generated_samples],
                          tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc,
                                     feed_dict=feed_dict_)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.max_seq_length - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards
