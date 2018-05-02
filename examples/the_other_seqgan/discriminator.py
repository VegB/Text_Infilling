"""
A RNN-Based Discriminator for SeqGAN.
"""
import tensorflow as tf
import texar as tx
from utils import *


class Discriminator:
    def __init__(self, config, word2id, class_num=2):
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)
        with tf.variable_scope('discriminator', initializer=initializer):
            self.batch_size = config.batch_size
            self.max_seq_length = config.num_steps
            self.vocab_size = len(word2id)
            self.class_num = class_num

            self.real_samples = tf.placeholder(dtype=tf.int32, name="samples",
                                          shape=[self.batch_size, self.max_seq_length + 1])
            self.gen_samples = tf.placeholder(dtype=tf.int32, name="samples",
                                               shape=[self.batch_size, self.max_seq_length + 1])

            self.embedder = tx.modules.WordEmbedder(
                vocab_size=self.vocab_size, hparams=config.emb)
            self.encoder = tx.modules.UnidirectionalRNNEncoder(
                hparams={"rnn_cell": config.d_cell})

            self.encoder_unit_num = config.cell["kwargs"]["num_units"]

            r_emb_inputs = self.embedder(self.real_samples)
            g_emb_inputs = self.embedder(self.gen_samples)
            if config.keep_prob < 1:
                r_emb_inputs = tf.nn.dropout(
                    r_emb_inputs, tx.utils.switch_dropout(config.keep_prob))
                g_emb_inputs = tf.nn.dropout(
                    g_emb_inputs, tx.utils.switch_dropout(config.keep_prob))
            r_enc_outputs, r_enc_last = self.encoder(inputs=r_emb_inputs)
            g_enc_outputs, g_enc_last = self.encoder(inputs=g_emb_inputs)

            # build classifying layer params
            self.W = tf.Variable(tf.random_uniform([self.encoder_unit_num, 1], -1.0, 1.0), name="W")
            r_preds = tf.einsum('ijk,kl->ijl', r_enc_outputs, self.W)
            self.r_preds = tf.sigmoid(tf.squeeze(r_preds, [2]))
            g_preds = tf.einsum('ijk,kl->ijl', g_enc_outputs, self.W)
            self.g_preds = tf.sigmoid(tf.squeeze(g_preds, [2]))

            eps = 1e-12
            r_loss = -tf.reduce_mean(tf.log(self.r_preds + eps))  # r_preds -> 1.
            f_loss = -tf.reduce_mean(tf.log(1 - self.g_preds + eps))  # g_preds -> 0.
            self.dis_loss = r_loss + f_loss

            self.global_step = tf.placeholder(tf.int32)
            self.train_op = tx.core.get_train_op(
                self.dis_loss, global_step=self.global_step, increment_global_step=False,
                hparams=config.d_opt)
