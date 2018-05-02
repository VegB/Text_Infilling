import tensorflow as tf
import texar as tx
from texar.losses.mle_losses import _mask_sequences
from utils import *


class Generator:
    def __init__(self, config, word2id, bos, eos, pad):
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)
        with tf.variable_scope('generator', initializer=initializer):
            self.batch_size = config.batch_size
            self.max_seq_length = config.num_steps
            self.vocab_size = len(word2id)
            self.bos_id = bos
            self.eos_id = eos
            self.pad_id = pad

            self.data_batch = tf.placeholder(dtype=tf.int32, name="data_batch",
                                             shape=[None, self.max_seq_length + 2])
            self.rewards = tf.placeholder(dtype=tf.float32, name='rewards',
                                          shape=[None, self.max_seq_length, 1])
            self.expected_reward = tf.Variable(tf.zeros((self.max_seq_length,)))

            self.embedder = tx.modules.WordEmbedder(
                vocab_size=self.vocab_size, hparams=config.emb)
            self.decoder = tx.modules.BasicRNNDecoder(
                vocab_size=self.vocab_size,
                hparams={"rnn_cell": config.cell,
                         "max_decoding_length_train": self.max_seq_length + 1,
                         "max_decoding_length_infer": self.max_seq_length})

            emb_inputs = self.embedder(self.data_batch[:, :-1])
            if config.keep_prob < 1:
                emb_inputs = tf.nn.dropout(
                    emb_inputs, tx.utils.switch_dropout(config.keep_prob))

            initial_state = self.decoder.zero_state(self.batch_size, tf.float32)
            self.outputs, final_state, seq_lengths = self.decoder(
                decoding_strategy="train_greedy",
                impute_finished=True,
                inputs=emb_inputs,
                sequence_length=[self.max_seq_length + 1] * self.batch_size,
                initial_state=initial_state)

            # teacher forcing
            self.teacher_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.outputs.logits,
                targets=self.data_batch[:, 1:],
                weights=tf.ones((self.batch_size, self.max_seq_length + 1))
            )

            self.global_step = tf.placeholder(tf.int32)
            self.train_op = tx.core.get_train_op(
                self.teacher_loss, global_step=self.global_step, increment_global_step=False,
                hparams=config.teacher_opt)

            # reward
            reward = self.rewards - self.expected_reward[:tf.shape(self.rewards)[1]]
            self.mean_reward = tf.reduce_mean(reward)
            exp_reward_loss = tf.reduce_mean(tf.abs(reward))
            self.exp_op = tx.core.get_train_op(
                exp_reward_loss, global_step=self.global_step, increment_global_step=False,
                hparams=config.reward_opt)

            # for generation
            self.generated_outputs, _, _ = self.decoder(
                decoding_strategy="infer_sample",
                start_tokens=[self.bos_id] * self.batch_size,
                end_token=self.eos_id,
                embedding=self.embedder,
                initial_state=initial_state)

            # gen loss
            reward = tf.expand_dims(tf.cumsum(reward, axis=1, reverse=True), -1)
            g_sequence = tf.one_hot(self.generated_outputs.sample_id, self.vocab_size)
            g_preds = tf.clip_by_value(self.generated_outputs.logits * g_sequence, 1e-20, 1)
            gen_reward = tf.log(g_preds) * reward
            self.gen_loss = -tf.reduce_mean(gen_reward)

            self.update_step = tf.placeholder(tf.int32)
            self.update_op = tx.core.get_train_op(
                self.gen_loss, global_step=self.update_step, increment_global_step=False,
                hparams=config.g_opt)
